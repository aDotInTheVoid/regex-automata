#[cfg(feature = "std")]
use core::fmt;
#[cfg(feature = "std")]
use core::iter;
use core::mem;
use core::slice;
#[cfg(feature = "std")]
use std::collections::BTreeSet;

#[cfg(feature = "std")]
use byteorder::{BigEndian, LittleEndian};
use byteorder::{ByteOrder, NativeEndian};
#[cfg(feature = "std")]
use regex_syntax::ParserBuilder;

use crate::classes::{Byte, ByteClasses, ByteSet};
use crate::dfa::automaton::{
    Automaton, Start, ALPHABET_LEN, EOF, MATCH_OFFSET,
};
#[cfg(feature = "std")]
use crate::dfa::determinize::Determinizer;
#[cfg(feature = "std")]
use crate::dfa::error::Error;
#[cfg(feature = "std")]
use crate::dfa::minimize::Minimizer;
#[cfg(feature = "std")]
use crate::dfa::sparse;
use crate::dfa::special::Special;
#[cfg(feature = "std")]
use crate::nfa::thompson;
#[cfg(feature = "std")]
use crate::state_id::write_state_id_bytes;
use crate::state_id::{dead_id, StateID};
use crate::MatchKind;

/// Masks used in serialization of DFAs.
pub(crate) const MASK_PREMULTIPLIED: u16 = 0b0000_0000_0000_0001;
pub(crate) const MASK_ANCHORED: u16 = 0b0000_0000_0000_0010;

/// The configuration used for compiling a dense DFA.
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    anchored: Option<bool>,
    minimize: Option<bool>,
    byte_classes: Option<bool>,
    match_kind: Option<MatchKind>,
    unicode_word_boundary: Option<bool>,
    quit: Option<ByteSet>,
}

impl Config {
    /// Return a new default dense DFA compiler configuration.
    pub fn new() -> Config {
        Config::default()
    }

    /// Set whether matching must be anchored at the beginning of the input.
    ///
    /// When enabled, a match must begin at the start of a search. When
    /// disabled, the DFA will act as if the pattern started with a `(?s:.)*?`,
    /// which enables a match to appear anywhere.
    ///
    /// By default this is disabled.
    ///
    /// **WARNING:** this is subtly different than using a `^` at the start of
    /// your regex. A `^` forces a regex to match exclusively at the start of
    /// input, regardless of where you start your search. In contrast, enabling
    /// this option will allow your regex to match anywhere in your input, but
    /// the match must start at the beginning of a search.
    ///
    /// For example, consider the haystack `aba` and the following searches:
    ///
    /// 1. The regex `^a` is compiled with `anchored=false` and searches
    ///    `aba` starting at position `2`. Since `^` requires the match to
    ///    start at the beginning of the input and `2 > 0`, no match is found.
    /// 2. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `2`. This reports a match at `[2, 3]` since
    ///    the match starts where the search started. Since there is no `^`,
    ///    there is no requirement for the match to start at the beginning of
    ///    the input.
    /// 3. The regex `a` is compiled with `anchored=true` and searches `aba`
    ///    starting at position `1`. Since `b` corresponds to position `1` and
    ///    since the regex is anchored, it finds no match.
    /// 4. The regex `a` is compiled with `anchored=false` and searches `aba`
    ///    startting at position `1`. Since the regex is neither anchored nor
    ///    starts with `^`, the regex is compiled with an implicit `(?s:.)*?`
    ///    prefix that permits it to match anywhere. Thus, it reports a match
    ///    at `[2, 3]`.
    pub fn anchored(mut self, yes: bool) -> Config {
        self.anchored = Some(yes);
        self
    }

    /// Minimize the DFA.
    ///
    /// When enabled, the DFA built will be minimized such that it is as small
    /// as possible.
    ///
    /// Whether one enables minimization or not depends on the types of costs
    /// you're willing to pay and how much you care about its benefits. In
    /// particular, minimization has worst case `O(n*k*logn)` time and `O(k*n)`
    /// space, where `n` is the number of DFA states and `k` is the alphabet
    /// size. In practice, minimization can be quite costly in terms of both
    /// space and time, so it should only be done if you're willing to wait
    /// longer to produce a DFA. In general, you might want a minimal DFA in
    /// the following circumstances:
    ///
    /// 1. You would like to optimize for the size of the automaton. This can
    ///    manifest in one of two ways. Firstly, if you're converting the
    ///    DFA into Rust code (or a table embedded in the code), then a minimal
    ///    DFA will translate into a corresponding reduction in code  size, and
    ///    thus, also the final compiled binary size. Secondly, if you are
    ///    building many DFAs and putting them on the heap, you'll be able to
    ///    fit more if they are smaller. Note though that building a minimal
    ///    DFA itself requires additional space; you only realize the space
    ///    savings once the minimal DFA is constructed (at which point, the
    ///    space used for minimization is freed).
    /// 2. You've observed that a smaller DFA results in faster match
    ///    performance. Naively, this isn't guaranteed since there is no
    ///    inherent difference between matching with a bigger-than-minimal
    ///    DFA and a minimal DFA. However, a smaller DFA may make use of your
    ///    CPU's cache more efficiently.
    /// 3. You are trying to establish an equivalence between regular
    ///    languages. The standard method for this is to build a minimal DFA
    ///    for each language and then compare them. If the DFAs are equivalent
    ///    (up to state renaming), then the languages are equivalent.
    ///
    /// Typically, minimization only makes sense as an offline process. That
    /// is, one might minimize a DFA before serializing it to persistent
    /// storage.
    ///
    /// This option is disabled by default.
    pub fn minimize(mut self, yes: bool) -> Config {
        self.minimize = Some(yes);
        self
    }

    /// Whether to attempt to shrink the size of the DFA's alphabet or not.
    ///
    /// This option is enabled by default and should never by disabled unless
    /// one is debugging a generated DFA.
    ///
    /// When enabled, each DFA will use a map from all possible bytes to their
    /// corresponding equivalence class. Each equivalence class represents a
    /// set of bytes that does not discriminate between a match and a non-match
    /// in the DFA. For example, the pattern `[ab]+` has at least two
    /// equivalence classes: a set containing `a` and `b` and a set containing
    /// every byte except for `a` and `b`. `a` and `b` are in the same
    /// equivalence classes because they never discriminate between a match
    /// and a non-match.
    ///
    /// The advantage of this map is that the size of the transition table can
    /// be reduced drastically from `#states * 256 * sizeof(id)` to `#states *
    /// k * sizeof(id)` where `k` is the number of equivalence classes. As a
    /// result, total space usage can decrease substantially. Moreover, since a
    /// smaller alphabet is used, DFA compilation becomes faster as well.
    ///
    /// **WARNING:** This is only useful for debugging DFAs. Disabling this
    /// does not yield any speed advantages. Namely, even when this is
    /// disabled, a byte class map is still used while searching. The only
    /// difference is that every byte will be forced into its own distinct
    /// equivalence class. This is useful for debugging the actual generated
    /// transitions because it lets one see the transitions defined on actual
    /// bytes instead of the equivalence classes.
    pub fn byte_classes(mut self, yes: bool) -> Config {
        self.byte_classes = Some(yes);
        self
    }

    /// Find the longest possible match.
    ///
    /// This is distinct from the default leftmost-first match semantics in
    /// that it treats all NFA states as having equivalent priority. In other
    /// words, the longest possible match is always found and it is not
    /// possible to implement non-greedy match semantics when this is set. That
    /// is, `a+` and `a+?` are equivalent when this is enabled.
    ///
    /// In particular, a practical issue with this option at the moment is that
    /// it prevents unanchored searches from working correctly, since
    /// unanchored searches are implemented by prepending an non-greedy `.*?`
    /// to the beginning of the pattern. As stated above, non-greedy match
    /// semantics aren't supported. Therefore, if this option is enabled and
    /// an unanchored search is requested, then building a DFA will return an
    /// error.
    ///
    /// This option is principally useful when building a reverse DFA for
    /// finding the start of a match. If you are building a regex with
    /// [`RegexBuilder`](struct.RegexBuilder.html), then this is handled for
    /// you automatically. The reason why this is necessary for start of match
    /// handling is because we want to find the earliest possible starting
    /// position of a match to satisfy leftmost-first match semantics. When
    /// matching in reverse, this means finding the longest possible match,
    /// hence, this option.
    ///
    /// By default this is disabled.
    pub fn match_kind(mut self, kind: MatchKind) -> Config {
        self.match_kind = Some(kind);
        self
    }

    /// Heuristically enable Unicode word boundaries.
    ///
    /// When set, this will attempt to implement Unicode word boundaries as if
    /// they were ASCII word boundaries. This only works when the search input
    /// is ASCII only. If a non-ASCII byte is observed while searching, then a
    /// [`NoMatch::Quit`](../../enum.NoMatch.html#variant.Quit)
    /// error is returned.
    ///
    /// Therefore, when enabling this option, callers _must_ be prepared
    /// to handle a `NoMatch` error during search. When using a
    /// [`Regex`](../struct.Regex.html), this corresponds to using the `try_`
    /// suite of methods. Alternatively, if callers can guarantee that their
    /// input is ASCII only, then a `NoMatch::Quit` error will never be
    /// returned while searching.
    ///
    /// If the regex pattern provided has no Unicode word boundary in it, then
    /// this option has no effect. (That is, quitting on a non-ASCII byte only
    /// occurs when this option is enabled _and_ a Unicode word boundary is
    /// present in the pattern.)
    ///
    /// This is almost equivalent to setting all non-ASCII bytes to be quit
    /// bytes. The only difference is that this will cause non-ASCII bytes to
    /// be quit bytes _only_ when a Unicode word boundary is present in the
    /// regex pattern.
    ///
    /// This is disabled by default.
    pub fn unicode_word_boundary(mut self, yes: bool) -> Config {
        // We have a separate option for this instead of just setting the
        // appropriate quit bytes here because we don't want to set quit bytes
        // for every regex. We only want to set them when the regex contains a
        // Unicode word boundary.
        self.unicode_word_boundary = Some(yes);
        self
    }

    /// Add a "quit" byte to the DFA.
    ///
    /// When a quit byte is seen during search time, then search will return
    /// a [`NoMatch::Quit`](../../enum.NoMatch.html#variant.Quit) error
    /// indicating the offset at which the search stopped.
    ///
    /// A quit byte will always overrule any other aspects of a regex. For
    /// example, if the `x` byte is added as a quit byte and the regex `\w` is
    /// used, then observing `x` will cause the search to quit immediately
    /// despite the fact that `x` is in the `\w` class.
    ///
    /// This mechanism is primarily useful for heuristically enabling certain
    /// features like Unicode word boundaries in a DFA. Namely, if the input
    /// to search is ASCII, then a Unicode word boundary can be implemented
    /// via an ASCII word boundary with no change in semantics. Thus, a DFA
    /// can attempt to match a Unicode word boundary but give up as soon as it
    /// observes a non-ASCII byte. Indeed, if callers set all non-ASCII bytes
    /// to be quit bytes, then Unicode word boundaries will be permitted when
    /// building DFAs.
    ///
    /// When enabling this option, callers _must_ be prepared to handle a
    /// `NoMatch` error during search. When using a
    /// [`Regex`](../struct.Regex.html), this corresponds to using the `try_`
    /// suite of methods.
    ///
    /// By default, there are no quit bytes set.
    ///
    /// # Panics
    ///
    /// This panics if Unicode word boundaries are enabled and any non-ASCII
    /// byte is removed from the set of quit bytes. Namely, enabling Unicode
    /// word boundaries requires setting every non-ASCII byte to a quit byte.
    /// So if the caller attempts to undo any of that, then this will panic.
    pub fn quit(mut self, byte: u8, yes: bool) -> Config {
        if self.get_unicode_word_boundary() && !byte.is_ascii() && !yes {
            panic!(
                "cannot set non-ASCII byte to be non-quit when \
                 Unicode word boundaries are enabled"
            );
        }
        if self.quit.is_none() {
            self.quit = Some(ByteSet::empty());
        }
        if yes {
            self.quit.as_mut().unwrap().add(byte);
        } else {
            self.quit.as_mut().unwrap().remove(byte);
        }
        self
    }

    pub fn get_anchored(&self) -> bool {
        self.anchored.unwrap_or(false)
    }

    pub fn get_minimize(&self) -> bool {
        self.minimize.unwrap_or(false)
    }

    pub fn get_byte_classes(&self) -> bool {
        self.byte_classes.unwrap_or(true)
    }

    pub fn get_match_kind(&self) -> MatchKind {
        self.match_kind.unwrap_or(MatchKind::LeftmostFirst)
    }

    pub fn get_unicode_word_boundary(&self) -> bool {
        self.unicode_word_boundary.unwrap_or(false)
    }

    pub fn get_quit(&self, byte: u8) -> bool {
        self.quit.map_or(false, |q| q.contains(byte))
    }

    pub(crate) fn overwrite(self, o: Config) -> Config {
        Config {
            anchored: o.anchored.or(self.anchored),
            minimize: o.minimize.or(self.minimize),
            byte_classes: o.byte_classes.or(self.byte_classes),
            match_kind: o.match_kind.or(self.match_kind),
            unicode_word_boundary: o
                .unicode_word_boundary
                .or(self.unicode_word_boundary),
            quit: o.quit.or(self.quit),
        }
    }
}

/// A builder for constructing a deterministic finite automaton from regular
/// expressions.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the generated DFA. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
///
/// This builder always constructs a *single* DFA. As such, this builder can
/// only be used to construct regexes that either detect the presence of a
/// match or find the end location of a match. A single DFA cannot produce both
/// the start and end of a match. For that information, use a
/// [`Regex`](struct.Regex.html), which can be similarly configured using
/// [`RegexBuilder`](struct.RegexBuilder.html).
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct Builder {
    config: Config,
    thompson: thompson::Builder,
}

#[cfg(feature = "std")]
impl Builder {
    /// Create a new dense DFA builder with the default configuration.
    pub fn new() -> Builder {
        Builder {
            config: Config::default(),
            thompson: thompson::Builder::new(),
        }
    }

    /// Build a DFA from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(
        &self,
        pattern: &str,
    ) -> Result<DFA<Vec<usize>, usize>, Error> {
        self.build_with_size::<usize>(pattern)
    }

    /// Build a DFA from the given pattern using a specific representation for
    /// the DFA's state IDs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    ///
    /// The representation of state IDs is determined by the `S` type
    /// parameter. In general, `S` is usually one of `u8`, `u16`, `u32`, `u64`
    /// or `usize`, where `usize` is the default used for `build`. The purpose
    /// of specifying a representation for state IDs is to reduce the memory
    /// footprint of a DFA.
    ///
    /// When using this routine, the chosen state ID representation will be
    /// used throughout determinization and minimization, if minimization
    /// was requested. Even if the minimized DFA can fit into the chosen
    /// state ID representation but the initial determinized DFA cannot,
    /// then this will still return an error. To get a minimized DFA with a
    /// smaller state ID representation, first build it with a bigger state ID
    /// representation, and then shrink the size of the DFA using one of its
    /// conversion routines, such as
    /// [`DFA::to_u16`](struct.DFA.html#method.to_u16).
    pub fn build_with_size<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<DFA<Vec<S>, S>, Error> {
        let nfa = self.thompson.build(pattern).map_err(Error::nfa)?;
        self.build_from_nfa_with_size(&nfa)
    }

    /// Build a DFA from the given NFA.
    pub fn build_from_nfa(
        &self,
        nfa: &thompson::NFA,
    ) -> Result<DFA<Vec<usize>, usize>, Error> {
        self.build_from_nfa_with_size::<usize>(nfa)
    }

    /// Build a DFA from the given NFA using a specific representation for
    /// the DFA's state IDs.
    pub fn build_from_nfa_with_size<S: StateID>(
        &self,
        nfa: &thompson::NFA,
    ) -> Result<DFA<Vec<S>, S>, Error> {
        let mut quit = self.config.quit.unwrap_or(ByteSet::empty());
        if self.config.get_unicode_word_boundary()
            && nfa.has_word_boundary_unicode()
        {
            for b in 0x80..=0xFF {
                quit.add(b);
            }
        }
        let classes = if self.config.get_byte_classes() {
            let mut set = nfa.byte_class_set().clone();
            if !quit.is_empty() {
                set.add_set(&quit);
            }
            set.byte_classes()
        } else {
            ByteClasses::singletons()
        };

        let mut dfa = DFA::empty(classes)?;
        Determinizer::new()
            .anchored(self.config.get_anchored())
            .match_kind(self.config.get_match_kind())
            .quit(quit)
            .run(nfa, &mut dfa)?;
        if self.config.get_minimize() {
            Minimizer::new(&mut dfa).run();
        }
        Ok(dfa)
    }

    /// Apply the given dense DFA configuration options to this builder.
    pub fn configure(&mut self, config: Config) -> &mut Builder {
        self.config = self.config.overwrite(config);
        self
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](../struct.SyntaxConfig.html).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    ///
    /// These settings only apply when constructing a DFA directly from a
    /// pattern.
    pub fn syntax(&mut self, config: crate::SyntaxConfig) -> &mut Builder {
        self.thompson.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](../nfa/thompson/struct.Config.html).
    ///
    /// This permits setting things like whether the DFA should match the regex
    /// in reverse or if additional time should be spent shrinking the size of
    /// the NFA.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut Builder {
        self.thompson.configure(config);
        self
    }
}

#[cfg(feature = "std")]
impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}

/// A dense table-based deterministic finite automaton (DFA).
///
/// A dense DFA represents the core matching primitive in this crate. That is,
/// logically, all DFAs have a single start state, one or more match states
/// and a transition table that maps the current state and the current byte of
/// input to the next state. A DFA can use this information to implement fast
/// searching. In particular, the use of a dense DFA generally makes the trade
/// off that match speed is the most valuable characteristic, even if building
/// the regex may take significant time *and* space. As such, the processing
/// of every byte of input is done with a small constant number of operations
/// that does not vary with the pattern, its size or the size of the alphabet.
/// If your needs don't line up with this trade off, then a dense DFA may not
/// be an adequate solution to your problem.
///
/// In contrast, a [sparse DFA](../sparse/struct.DFA.html) makes the opposite
/// trade off: it uses less space but will execute a variable number of
/// instructions per byte at match time, which makes it slower for matching.
///
/// A DFA can be built using the default configuration via the
/// [`DFA::new`](struct.DFA.html#method.new) constructor. Otherwise,
/// one can configure various aspects via the
/// [`dense::Builder`](dense/struct.Builder.html).
///
/// A single DFA fundamentally supports the following operations:
///
/// 1. Detection of a match.
/// 2. Location of the end of the first possible match.
/// 3. Location of the end of the leftmost-first match.
///
/// A notable absence from the above list of capabilities is the location of
/// the *start* of a match. In order to provide both the start and end of a
/// match, *two* DFAs are required. This functionality is provided by a
/// [`Regex`](struct.Regex.html), which can be built with its basic
/// constructor, [`Regex::new`](struct.Regex.html#method.new), or with
/// a [`RegexBuilder`](struct.RegexBuilder.html).
///
/// # State size
///
/// A `DFA` has two type parameters, `T` and `S`. `T` corresponds to
/// the type of the DFA's transition table while `S` corresponds to the
/// representation used for the DFA's state identifiers as described by the
/// [`StateID`](trait.StateID.html) trait. This type parameter is typically
/// `usize`, but other valid choices provided by this crate include `u8`,
/// `u16`, `u32` and `u64`. The primary reason for choosing a different state
/// identifier representation than the default is to reduce the amount of
/// memory used by a DFA. Note though, that if the chosen representation cannot
/// accommodate the size of your DFA, then building the DFA will fail and
/// return an error.
///
/// While the reduction in heap memory used by a DFA is one reason for choosing
/// a smaller state identifier representation, another possible reason is for
/// decreasing the serialization size of a DFA, as returned by
/// [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian),
/// [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
/// or
/// [`to_bytes_native_endian`](struct.DFA.html#method.to_bytes_native_endian).
///
/// The type of the transition table is typically either `Vec<S>` or `&[S]`,
/// depending on where the transition table is stored.
///
/// # Variants
///
/// This DFA is defined as a non-exhaustive enumeration of different types of
/// dense DFAs. All of these dense DFAs use the same internal representation
/// for the transition table, but they vary in how the transition table is
/// read. A DFA's specific variant depends on the configuration options set via
/// [`dense::Builder`](dense/struct.Builder.html). The default variant is
/// `PremultipliedByteClass`.
///
/// # The `Automaton` trait
///
/// This type implements the [`Automaton`](trait.Automaton.html) trait, which
/// means it can be used for searching. For example:
///
/// ```
/// use regex_automata::dfa::{Automaton, dense};
///
/// # fn example() -> Result<(), regex_automata::dfa::Error> {
/// let dfa = dense::DFA::new("foo[0-9]+")?;
/// assert_eq!(Ok(Some(8)), dfa.find_leftmost_fwd(b"foo12345"));
/// # Ok(()) }; example().unwrap()
/// ```
#[derive(Clone)]
pub struct DFA<T, S = usize> {
    /// The total number of states in this DFA. Note that a DFA always has at
    /// least one state---the dead state---even the empty DFA. In particular,
    /// the dead state always has ID 0 and is correspondingly always the first
    /// state. The dead state is never a match state.
    state_count: usize,
    /// Information about which states as "special." Special states are states
    /// that are dead, quit, matching, starting or accelerated. For more info,
    /// see the docs for `Special`.
    special: Special<S>,
    /// A set of equivalence classes, where a single equivalence class
    /// represents a set of bytes that never discriminate between a match
    /// and a non-match in the DFA. Each equivalence class corresponds to
    /// a single letter in this DFA's alphabet, where the maximum number of
    /// letters is 256 (each possible value of a byte). Consequently, the
    /// number of equivalence classes corresponds to the number of transitions
    /// for each DFA state.
    ///
    /// The only time the number of equivalence classes is fewer than 256 is
    /// if the DFA's kind uses byte classes. If the DFA doesn't use byte
    /// classes, then this vector is empty.
    byte_classes: ByteClasses,
    /// The initial start state IDs.
    ///
    /// In practice, T is either Vec<S> or &[S].
    starts: T,
    /// A contiguous region of memory representing the transition table in
    /// row-major order. The representation is dense. That is, every state has
    /// precisely the same number of transitions. The maximum number of
    /// transitions is 256. If a DFA has been instructed to use byte classes,
    /// then the number of transitions can be much less.
    ///
    /// In practice, T is either Vec<S> or &[S].
    trans: T,
}

#[cfg(feature = "std")]
impl DFA<Vec<usize>, usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding DFA.
    ///
    /// The default configuration uses `usize` for state IDs. The DFA is *not*
    /// minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`dense::Builder`](dense/struct.Builder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa = dense::DFA::new("foo[0-9]+bar")?;
    /// assert_eq!(Ok(Some(11)), dfa.find_leftmost_fwd(b"foo12345bar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new(pattern: &str) -> Result<DFA<Vec<usize>, usize>, Error> {
        Builder::new().build(pattern)
    }
}

#[cfg(feature = "std")]
impl<S: StateID> DFA<Vec<S>, S> {
    /// Create a new DFA that matches every input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that always matches, callers must provide a
    /// type hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa: dense::DFA<Vec<usize>, usize> = dense::DFA::always_match()?;
    /// assert_eq!(Ok(Some(0)), dfa.find_leftmost_fwd(b""));
    /// assert_eq!(Ok(Some(0)), dfa.find_leftmost_fwd(b"foo"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn always_match() -> Result<DFA<Vec<S>, S>, Error> {
        let nfa = thompson::NFA::always_match();
        Builder::new().build_from_nfa_with_size(&nfa)
    }

    /// Create a new DFA that never matches any input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that never matches, callers must provide a type
    /// hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa: dense::DFA<Vec<usize>, usize> = dense::DFA::never_match()?;
    /// assert_eq!(Ok(None), dfa.find_leftmost_fwd(b""));
    /// assert_eq!(Ok(None), dfa.find_leftmost_fwd(b"foo"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn never_match() -> Result<DFA<Vec<S>, S>, Error> {
        let nfa = thompson::NFA::never_match();
        Builder::new().build_from_nfa_with_size(&nfa)
    }

    /// Create a new DFA with the given set of byte equivalence classes. The
    /// DFA contains a single dead state that never matches any input.
    fn empty(byte_classes: ByteClasses) -> Result<DFA<Vec<S>, S>, Error> {
        let mut dfa = DFA {
            state_count: 0,
            special: Special::new(),
            byte_classes,
            starts: vec![dead_id(); Start::count()],
            trans: vec![],
        };
        // dead state
        dfa.add_empty_state()?;
        // quit state
        dfa.add_empty_state()?;
        Ok(dfa)
    }
}

impl<T: AsRef<[S]>, S: StateID> DFA<T, S> {
    /// Cheaply return a borrowed version of this dense DFA. Specifically, the
    /// DFA returned always uses `&[S]` for its transition table while keeping
    /// the same state identifier representation.
    pub fn as_ref<'a>(&'a self) -> DFA<&'a [S], S> {
        DFA {
            state_count: self.state_count,
            special: self.special,
            byte_classes: self.byte_classes().clone(),
            starts: self.starts(),
            trans: self.trans(),
        }
    }

    /// Return an owned version of this sparse DFA. Specifically, the DFA
    /// returned always uses `Vec<u8>` for its transition table while keeping
    /// the same state identifier representation.
    ///
    /// Effectively, this returns a sparse DFA whose transition table lives
    /// on the heap.
    #[cfg(feature = "std")]
    pub fn to_owned(&self) -> DFA<Vec<S>, S> {
        DFA {
            state_count: self.state_count,
            special: self.special,
            byte_classes: self.byte_classes().clone(),
            starts: self.starts().to_vec(),
            trans: self.trans().to_vec(),
        }
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This corresponds to heap memory
    /// usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<dense::DFA>()`.
    pub fn memory_usage(&self) -> usize {
        self.trans().len() * mem::size_of::<S>()
    }
}

/// Routines for converting a dense DFA to other representations, such as
/// sparse DFAs, smaller state identifiers or raw bytes suitable for persistent
/// storage.
#[cfg(feature = "std")]
impl<T: AsRef<[S]>, S: StateID> DFA<T, S> {
    /// Convert this dense DFA to a sparse DFA.
    ///
    /// This is a convenience routine for `to_sparse_sized` that fixes the
    /// state identifier representation of the sparse DFA to the same
    /// representation used for this dense DFA.
    ///
    /// If the chosen state identifier representation is too small to represent
    /// all states in the sparse DFA, then this returns an error. In most
    /// cases, if a dense DFA is constructable with `S` then a sparse DFA will
    /// be as well. However, it is not guaranteed.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dense = dense::DFA::new("foo[0-9]+")?;
    /// let sparse = dense.to_sparse()?;
    /// assert_eq!(Ok(Some(8)), sparse.find_leftmost_fwd(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn to_sparse(&self) -> Result<sparse::DFA<Vec<u8>, S>, Error> {
        self.to_sparse_sized()
    }

    /// Convert this dense DFA to a sparse DFA.
    ///
    /// Using this routine requires supplying a type hint to choose the state
    /// identifier representation for the resulting sparse DFA.
    ///
    /// If the chosen state identifier representation is too small to represent
    /// all states in the sparse DFA, then this returns an error.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dense = dense::DFA::new("foo[0-9]+")?;
    /// let sparse = dense.to_sparse_sized::<u16>()?;
    /// assert_eq!(Ok(Some(8)), sparse.find_leftmost_fwd(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn to_sparse_sized<A: StateID>(
        &self,
    ) -> Result<sparse::DFA<Vec<u8>, A>, Error> {
        sparse::DFA::from_dense_sized(self)
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u8` for the representation of state identifiers.
    /// If `u8` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u8>()`.
    pub fn to_u8(&self) -> Result<DFA<Vec<u8>, u8>, Error> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u16` for the representation of state identifiers.
    /// If `u16` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u16>()`.
    pub fn to_u16(&self) -> Result<DFA<Vec<u16>, u16>, Error> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u32` for the representation of state identifiers.
    /// If `u32` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u32>()`.
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    pub fn to_u32(&self) -> Result<DFA<Vec<u32>, u32>, Error> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA,
    /// but attempt to use `u64` for the representation of state identifiers.
    /// If `u64` is insufficient to represent all state identifiers in this
    /// DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u64>()`.
    #[cfg(target_pointer_width = "64")]
    pub fn to_u64(&self) -> Result<DFA<Vec<u64>, u64>, Error> {
        self.to_sized()
    }

    /// Create a new DFA whose match semantics are equivalent to this DFA, but
    /// attempt to use `A` for the representation of state identifiers. If `A`
    /// is insufficient to represent all state identifiers in this DFA, then
    /// this returns an error.
    ///
    /// An alternative way to construct such a DFA is to use
    /// [`dense::Builder::build_with_size`](dense/struct.Builder.html#method.build_with_size).
    /// In general, using the builder is preferred since it will use the given
    /// state identifier representation throughout determinization (and
    /// minimization, if done), and thereby using less memory throughout the
    /// entire construction process. However, these routines are necessary
    /// in cases where, say, a minimized DFA could fit in a smaller state
    /// identifier representation, but the initial determinized DFA would not.
    pub fn to_sized<A: StateID>(&self) -> Result<DFA<Vec<A>, A>, Error> {
        // Check that this DFA can fit into A's representation.
        let last_state_id = (self.state_count - 1) * self.alphabet_len();
        if last_state_id > A::max_id() {
            return Err(Error::state_id_overflow(A::max_id()));
        }

        // We're off to the races. The new DFA is the same as the old one,
        // except all state IDs are represented by `A` instead of `S`.
        let mut new = DFA {
            state_count: self.state_count,
            special: self.special.to_sized()?,
            byte_classes: self.byte_classes().clone(),
            starts: vec![dead_id::<A>(); self.starts().len()],
            trans: vec![dead_id::<A>(); self.trans().len()],
        };
        for (i, id) in new.starts.iter_mut().enumerate() {
            *id = A::from_usize(self.starts()[i].as_usize());
        }
        for (i, id) in new.trans.iter_mut().enumerate() {
            *id = A::from_usize(self.trans()[i].as_usize());
        }
        Ok(new)
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary, in little
    /// endian format.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_little_endian(&self) -> Result<Vec<u8>, Error> {
        self.to_bytes::<LittleEndian>()
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary, in big
    /// endian format.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_big_endian(&self) -> Result<Vec<u8>, Error> {
        self.to_bytes::<BigEndian>()
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary, in native
    /// endian format. Generally, it is better to pick an explicit endianness
    /// using either `to_bytes_little_endian` or `to_bytes_big_endian`. This
    /// routine is useful in tests where the DFA is serialized and deserialized
    /// on the same platform.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_native_endian(&self) -> Result<Vec<u8>, Error> {
        self.to_bytes::<NativeEndian>()
    }

    /// Serialize a DFA to raw bytes, aligned to an 8 byte boundary.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    fn to_bytes<A: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let label = b"rust-regex-automata-dfa\x00";
        assert_eq!(24, label.len());

        let special_size = <Special<S>>::num_bytes();
        assert_eq!(
            0,
            special_size % mem::size_of::<S>(),
            "special size is not a multiple of state ID representation size",
        );
        let starts_size = mem::size_of::<S>() * self.starts().len();
        let trans_size = mem::size_of::<S>() * self.trans().len();
        let size =
            // For human readable label.
            label.len()
            // endiannes check, must be equal to 0xFEFF for native endian
            + 2
            // For version number.
            + 2
            // Size of state ID representation, in bytes.
            // Must be 1, 2, 4 or 8.
            + 2
            // For DFA misc options.
            + 2
            // For state count.
            + 8
            // For start state count.
            + 8
            // For byte class map.
            + 256
            // For special state ID info.
            + special_size
            // For start state IDs.
            + starts_size
            // For transition table.
            + trans_size;
        // sanity check, this can be updated if need be
        assert_eq!(304 + special_size + starts_size + trans_size, size);
        // This must always pass. It checks that the start state and transition
        // tables are at a properly aligned address.
        assert_eq!(
            0,
            (size - (starts_size + trans_size)) % mem::align_of::<S>()
        );
        assert_eq!(0, (size - trans_size) % mem::align_of::<S>());

        let mut buf = vec![0; size];
        let mut i = 0;

        // write label
        for &b in label {
            buf[i] = b;
            i += 1;
        }
        // endianness check
        A::write_u16(&mut buf[i..], 0xFEFF);
        i += 2;
        // version number
        A::write_u16(&mut buf[i..], 2);
        i += 2;
        // size of state ID
        let state_size = mem::size_of::<S>();
        if ![1, 2, 4, 8].contains(&state_size) {
            return Err(Error::serialize(&format!(
                "state size of {} not supported, must be 1, 2, 4 or 8",
                state_size
            )));
        }
        A::write_u16(&mut buf[i..], state_size as u16);
        i += 2;
        // DFA misc options (currently unused)
        let mut options = 0u16;
        A::write_u16(&mut buf[i..], options);
        i += 2;
        // state count
        A::write_u64(&mut buf[i..], self.state_count as u64);
        i += 8;
        // start state count
        A::write_u64(&mut buf[i..], self.starts().len() as u64);
        i += 8;
        // byte class map
        for b in (0..256).map(|b| b as u8) {
            buf[i] = self.byte_classes().get(b);
            i += 1;
        }
        // special state ID info
        self.special.to_bytes::<A>(&mut buf[i..]);
        i += <Special<S>>::num_bytes();
        // start state IDs
        for &id in self.starts() {
            write_state_id_bytes::<A, _>(&mut buf[i..], id);
            i += state_size;
        }
        // transition table
        for &id in self.trans() {
            write_state_id_bytes::<A, _>(&mut buf[i..], id);
            i += state_size;
        }
        assert_eq!(size, i, "expected to consume entire buffer");

        Ok(buf)
    }
}

impl<'a, S: StateID> DFA<&'a [S], S> {
    /// Deserialize a DFA with a specific state identifier representation.
    ///
    /// Deserializing a DFA using this routine will never allocate heap memory.
    /// This is also guaranteed to be a constant time operation that does not
    /// vary with the size of the DFA.
    ///
    /// The bytes given should be generated by the serialization of a DFA with
    /// either the
    /// [`to_bytes_little_endian`](struct.DFA.html#method.to_bytes_little_endian)
    /// method or the
    /// [`to_bytes_big_endian`](struct.DFA.html#method.to_bytes_big_endian)
    /// endian, depending on the endianness of the machine you are
    /// deserializing this DFA from.
    ///
    /// If the state identifier representation is `usize`, then deserialization
    /// is dependent on the pointer size. For this reason, it is best to
    /// serialize DFAs using a fixed size representation for your state
    /// identifiers, such as `u8`, `u16`, `u32` or `u64`.
    ///
    /// # Panics
    ///
    /// The bytes given should be *trusted*. In particular, if the bytes
    /// are not a valid serialization of a DFA, or if the given bytes are
    /// not aligned to an 8 byte boundary, or if the endianness of the
    /// serialized bytes is different than the endianness of the machine that
    /// is deserializing the DFA, then this routine will panic. Moreover, it is
    /// possible for this deserialization routine to succeed even if the given
    /// bytes do not represent a valid serialized dense DFA.
    ///
    /// # Safety
    ///
    /// This routine is unsafe because it permits callers to provide an
    /// arbitrary transition table with possibly incorrect transitions. While
    /// the various serialization routines will never return an incorrect
    /// transition table, there is no guarantee that the bytes provided here
    /// are correct. While deserialization does many checks (as documented
    /// above in the panic conditions), this routine does not check that the
    /// transition table is correct. Given an incorrect transition table, it is
    /// possible for the search routines to access out-of-bounds memory because
    /// of explicit bounds check elision.
    ///
    /// # Example
    ///
    /// This example shows how to serialize a DFA to raw bytes, deserialize it
    /// and then use it for searching. Note that we first convert the DFA to
    /// using `u16` for its state identifier representation before serializing
    /// it. While this isn't strictly necessary, it's good practice in order to
    /// decrease the size of the DFA and to avoid platform specific pitfalls
    /// such as differing pointer sizes.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let initial = dense::DFA::new("foo[0-9]+")?;
    /// let bytes = initial.to_u16()?.to_bytes_native_endian()?;
    /// let dfa: dense::DFA<&[u16], u16> = unsafe {
    ///     dense::DFA::from_bytes_unchecked(&bytes)
    /// };
    ///
    /// assert_eq!(Ok(Some(8)), dfa.find_leftmost_fwd(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub unsafe fn from_bytes_unchecked(mut buf: &'a [u8]) -> DFA<&'a [S], S> {
        assert_eq!(
            0,
            buf.as_ptr() as usize % mem::align_of::<S>(),
            "dense::DFA starting at address {} is not aligned to {} bytes",
            buf.as_ptr() as usize,
            mem::align_of::<S>()
        );

        // skip over label
        match buf.iter().position(|&b| b == b'\x00') {
            None => panic!("could not find label"),
            Some(i) => buf = &buf[i + 1..],
        }

        // check that current endianness is same as endianness of DFA
        let endian_check = NativeEndian::read_u16(buf);
        buf = &buf[2..];
        if endian_check != 0xFEFF {
            panic!(
                "endianness mismatch, expected 0xFEFF but got 0x{:X}. \
                 are you trying to load a dense::DFA serialized with a \
                 different endianness?",
                endian_check,
            );
        }

        // check that the version number is supported
        let version = NativeEndian::read_u16(buf);
        buf = &buf[2..];
        if version != 2 {
            panic!(
                "expected version 2, but found unsupported version {}",
                version,
            );
        }

        // read size of state
        let state_size = NativeEndian::read_u16(buf) as usize;
        if state_size != mem::size_of::<S>() {
            panic!(
                "state size of dense::DFA ({}) does not match \
                 requested state size ({})",
                state_size,
                mem::size_of::<S>(),
            );
        }
        buf = &buf[2..];

        // read miscellaneous options (currently unused)
        let _opts = NativeEndian::read_u16(buf);
        buf = &buf[2..];

        // read state count
        let state_count = NativeEndian::read_u64(buf) as usize;
        buf = &buf[8..];

        // read start state count
        let start_len = NativeEndian::read_u64(buf) as usize;
        buf = &buf[8..];

        // read byte classes
        let byte_classes = ByteClasses::from_slice(&buf[..256]);
        buf = &buf[256..];

        // read special state ID info.
        let special = Special::from_bytes(&buf).unwrap();
        buf = &buf[<Special<S>>::num_bytes()..];

        // read start state IDs
        let start_len_bytes = start_len * state_size;
        assert!(
            buf.len() >= start_len_bytes,
            "insufficient start state table bytes, \
             expected at least {} but only have {}",
            start_len_bytes,
            buf.len(),
        );
        assert_eq!(
            0,
            buf.as_ptr() as usize % mem::align_of::<S>(),
            "dense::DFA start state table is not properly aligned",
        );
        // SAFETY: The key things we need to worry about here are alignment and
        // size. The two asserts above should cover both conditions.
        let starts =
            slice::from_raw_parts(buf.as_ptr() as *const S, start_len);
        buf = &buf[start_len_bytes..];

        let len = state_count * byte_classes.alphabet_len();
        let len_bytes = len * state_size;
        assert_eq!(
            buf.len(),
            len_bytes,
            "incorrect number of transition table bytes, \
             expected {} but have {}",
            len_bytes,
            buf.len()
        );
        assert_eq!(
            0,
            buf.as_ptr() as usize % mem::align_of::<S>(),
            "dense::DFA transition table is not properly aligned"
        );

        // SAFETY: The key things we need to worry about here are alignment and
        // size. The two asserts above should cover both conditions.
        let trans = slice::from_raw_parts(buf.as_ptr() as *const S, len);
        DFA { starts, state_count, special, byte_classes, trans }
    }
}

/// The following methods implement mutable routines on the internal
/// representation of a DFA. As such, we must fix the first type parameter to
/// a `Vec<S>` since a generic `T: AsRef<[S]>` does not permit mutation. We
/// can get away with this because these methods are internal to the crate and
/// are exclusively used during construction of the DFA.
#[cfg(feature = "std")]
impl<S: StateID> DFA<Vec<S>, S> {
    /// Add a start state of this DFA.
    pub(crate) fn set_start_state(&mut self, index: Start, id: S) {
        assert!(self.to_index(id) < self.state_count, "invalid start state");

        self.starts[index.as_usize()] = id;
    }

    /// Add the given transition to this DFA. Both the `from` and `to` states
    /// must already exist.
    pub(crate) fn add_transition(&mut self, from: S, byte: Byte, to: S) {
        assert!(self.to_index(from) < self.state_count, "invalid from state");
        assert!(self.to_index(to) < self.state_count, "invalid to state");

        let class = match byte {
            Byte::U8(b) => self.byte_classes().get(b) as usize,
            Byte::EOF(b) => b as usize,
        };
        self.trans[from.as_usize() + class] = to;
    }

    /// An an empty state (a state where all transitions lead to a dead state)
    /// and return its identifier. The identifier returned is guaranteed to
    /// not point to any other existing state.
    ///
    /// If adding a state would exhaust the state identifier space (given by
    /// `S`), then this returns an error. In practice, this means that the
    /// state identifier representation chosen is too small.
    pub(crate) fn add_empty_state(&mut self) -> Result<S, Error> {
        let alpha_len = self.alphabet_len();
        let id = if self.state_count == 0 {
            S::from_usize(0)
        } else {
            // Normally, to get a fresh state identifier, we would just
            // take the index of the next state added to the transition
            // table. However, we actually perform an optimization here that
            // premultiplies state IDs by the alphabet length, such that
            // they point immediately at the beginning of their transitions
            // in the transition table. This avoids an extra multiplication
            // instruction for state lookup at search time.
            //
            // Premultiplied identifiers means that instead of your matching
            // loop looking something like this:
            //
            //   state = dfa.start
            //   for byte in haystack:
            //       next = dfa.transitions[state * len(alphabet) + byte]
            //       if dfa.is_match(next):
            //           return true
            //   return false
            //
            // it can instead look like this:
            //
            //   state = dfa.start
            //   for byte in haystack:
            //       next = dfa.transitions[state + byte]
            //       if dfa.is_match(next):
            //           return true
            //   return false
            //
            // In other words, we save a multiplication instruction in the
            // critical path. This turns out to be a decent performance win.
            // The cost of using premultiplied state ids is that they can
            // require a bigger state id representation. (And they also make
            // the code a bit more complex, especially during minimization and
            // when reshuffling states, as one needs to convert back and forth
            // between state IDs and state indices.)
            let next = match self.state_count.checked_mul(alpha_len) {
                Some(next) => next,
                None => return Err(Error::state_id_overflow(std::usize::MAX)),
            };
            if next > S::max_id() {
                return Err(Error::state_id_overflow(S::max_id()));
            }
            S::from_usize(next)
        };
        self.trans.extend(iter::repeat(dead_id::<S>()).take(alpha_len));
        // This should never panic, since state_count is a usize. The
        // transition table size would have run out of room long ago.
        self.state_count = self.state_count.checked_add(1).unwrap();
        Ok(id)
    }

    /// Return a mutable representation of the state corresponding to the given
    /// id. This is useful for implementing routines that manipulate DFA states
    /// (e.g., swapping states).
    pub(crate) fn get_state_mut(&mut self, id: S) -> StateMut<'_, S> {
        let alphabet_len = self.alphabet_len();
        let i = id.as_usize();
        StateMut { transitions: &mut self.trans[i..i + alphabet_len] }
    }

    /// Swap the two states given in the transition table.
    ///
    /// This routine does not do anything to check the correctness of this
    /// swap. Callers must ensure that other states pointing to id1 and id2 are
    /// updated appropriately.
    pub(crate) fn swap_states(&mut self, id1: S, id2: S) {
        for b in 0..self.alphabet_len() {
            self.trans.swap(id1.as_usize() + b, id2.as_usize() + b);
        }
    }

    /// Truncate the states in this DFA to the given count.
    ///
    /// This routine does not do anything to check the correctness of this
    /// truncation. Callers must ensure that other states pointing to truncated
    /// states are updated appropriately.
    pub(crate) fn truncate_states(&mut self, count: usize) {
        let alphabet_len = self.alphabet_len();
        self.trans.truncate(count * alphabet_len);
        self.state_count = count;
    }

    pub(crate) fn shuffle(&mut self, is_match: &BTreeSet<S>) {
        // The determinizer always adds a quit state and it is always second.
        self.special.quit_id = self.from_index(1);
        // If all we have are the dead and quit states, then we're done.
        if self.state_count <= 2 {
            return;
        }

        // Collect all our start states into a convenient set and confirm there
        // is no overlap with match states. In the classicl DFA construction,
        // start states can be match states. But because of look-around, we
        // delay all matches by a byte, which prevents start states from being
        // match states.
        let is_start: BTreeSet<S> = self.starts().iter().cloned().collect();
        assert!(
            is_match.intersection(&is_start).next().is_none(),
            "matching and starting states must not overlap",
        );

        // We implement shuffling by a sequence of pairwise swaps of states.
        // Since we have a number of things referencing states via their IDs
        // and swapping them changes their IDs, we need to record every swap
        // we make so that we can remap IDs. We start off with the identity
        // map and swap them as appropriate.
        let mut remap: Vec<S> =
            (0..self.state_count).map(|i| self.from_index(i)).collect();

        if is_match.is_empty() {
            self.special.min_match = dead_id();
            self.special.max_match = dead_id();
        } else {
            // The determinizer guarantees that the first two states are the
            // dead and quit states, respectively. We want our match states to
            // come right after quit. So find the first available ID, which
            // corresponds to the first non-match state after the quit state.
            let max_id = self.from_index(self.state_count - 1);
            let mut next_id = self.from_index(2);
            while next_id < max_id && is_match.contains(&next_id) {
                next_id = self.next_state_id(next_id);
            }
            self.special.min_match = next_id;
            // We now start at the end of the DFA and work our way backwards.
            // When we see a match state, we swap it with our available ID and
            // then find the next available ID. Rinse and repeat until we meet
            // in the middle.
            let mut cur_id = max_id;
            while cur_id > next_id {
                if is_match.contains(&cur_id) {
                    self.swap_states(cur_id, next_id);
                    remap.swap(self.to_index(cur_id), self.to_index(next_id));

                    next_id = self.next_state_id(next_id);
                    while next_id < cur_id && is_match.contains(&next_id) {
                        next_id = self.next_state_id(next_id);
                    }
                }
                cur_id = self.prev_state_id(cur_id);
            }
            self.special.max_match = self.prev_state_id(next_id);
        }

        // Now that we've finished shuffling, we need to remap all of our
        // transitions.
        let alphabet_len = self.alphabet_len();
        // To work around the borrow checker. Cannot borrow self while mutably
        // iterating over a state's transitions.
        let to_index = |id: S| -> usize { id.as_usize() / alphabet_len };
        for &id in remap.iter() {
            for (_, next_id) in self.get_state_mut(id).iter_mut() {
                let i = to_index(*next_id);
                *next_id = remap[i];
            }
        }
        for start_id in self.starts.iter_mut() {
            let i = to_index(*start_id);
            *start_id = remap[i];
        }

        self.special.set_max();
    }
}

/// A variety of generic internal methods for accessing DFA internals.
impl<T: AsRef<[S]>, S: StateID> DFA<T, S> {
    /// Return the byte classes used by this DFA.
    pub(crate) fn byte_classes(&self) -> &ByteClasses {
        &self.byte_classes
    }

    /// Return the info about special states.
    pub(crate) fn special(&self) -> &Special<S> {
        &self.special
    }

    /// Return the info about special states as a mutable borrow.
    pub(crate) fn special_mut(&mut self) -> &mut Special<S> {
        &mut self.special
    }

    /// Returns an iterator over all states in this DFA.
    ///
    /// This iterator yields a tuple for each state. The first element of the
    /// tuple corresponds to a state's identifier, and the second element
    /// corresponds to the state itself (comprised of its transitions).
    #[cfg(feature = "std")]
    pub(crate) fn states(&self) -> StateIter<'_, T, S> {
        let it = self.trans().chunks(self.alphabet_len());
        StateIter { dfa: self, it: it.enumerate() }
    }

    /// Return the total number of states in this DFA. Every DFA has at least
    /// 1 state, even the empty DFA.
    #[cfg(feature = "std")]
    pub(crate) fn state_count(&self) -> usize {
        self.state_count
    }

    /// Return the number of elements in this DFA's alphabet.
    ///
    /// If this DFA doesn't use byte classes, then this is always equivalent
    /// to 256. Otherwise, it is guaranteed to be some value less than or equal
    /// to 256.
    pub(crate) fn alphabet_len(&self) -> usize {
        self.byte_classes().alphabet_len()
    }

    /// Return the state IDs for this DFA's start states.
    pub(crate) fn starts(&self) -> &[S] {
        self.starts.as_ref()
    }

    /// Returns the ID of the dead state for this DFA.
    pub(crate) fn dead_id(&self) -> S {
        self.from_index(0)
    }

    /// Returns the ID of the quit state for this DFA.
    pub(crate) fn quit_id(&self) -> S {
        self.from_index(1)
    }

    /// Convert the given state identifier to the state's index. The state's
    /// index corresponds to the position in which it appears in the transition
    /// table. When a DFA is NOT premultiplied, then a state's identifier is
    /// also its index. When a DFA is premultiplied, then a state's identifier
    /// is equal to `index * alphabet_len`. This routine reverses that.
    pub(crate) fn to_index(&self, id: S) -> usize {
        id.as_usize() / self.alphabet_len()
    }

    pub(crate) fn from_index(&self, index: usize) -> S {
        S::from_usize(index * self.alphabet_len())
    }

    /// Returns the state ID for the state immediately following the one given.
    ///
    /// This does not check whether the state ID returned is invalid. In fact,
    /// if the state ID given is the last state in this DFA, then the state ID
    /// returned is guaranteed to be invalid.
    pub(crate) fn next_state_id(&self, id: S) -> S {
        self.from_index(self.to_index(id).checked_add(1).unwrap())
    }

    /// Returns the state ID for the state immediately preceding the one given.
    ///
    /// If the dead ID given (which is zero), then this panics.
    pub(crate) fn prev_state_id(&self, id: S) -> S {
        self.from_index(self.to_index(id).checked_sub(1).unwrap())
    }

    /// Return this DFA's transition table as a slice.
    fn trans(&self) -> &[S] {
        self.trans.as_ref()
    }
}

#[cfg(feature = "std")]
impl<T: AsRef<[S]>, S: StateID> fmt::Debug for DFA<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn state_status<T: AsRef<[S]>, S: StateID>(
            dfa: &DFA<T, S>,
            id: S,
        ) -> &'static str {
            if dfa.is_dead_state(id) {
                "D "
            } else if dfa.is_quit_state(id) {
                "Q "
            } else if dfa.starts().contains(&id) {
                if dfa.is_match_state(id) {
                    ">*"
                } else {
                    "> "
                }
            } else {
                if dfa.is_match_state(id) {
                    " *"
                } else {
                    "  "
                }
            }
        }

        writeln!(f, "dense::DFA(")?;
        for (id, state) in self.states() {
            let status = state_status(self, id);
            let id =
                if f.alternate() { id.as_usize() } else { self.to_index(id) };
            write!(f, "{}{:06}: ", status, id)?;
            state.fmt(f)?;
            write!(f, "\n")?;
        }
        writeln!(f, "")?;
        for (i, &start_id) in self.starts().iter().enumerate() {
            let id = if f.alternate() {
                start_id.as_usize()
            } else {
                self.to_index(start_id)
            };
            writeln!(
                f,
                "START({}): {:?} => {:06}",
                i,
                Start::from_usize(i),
                id,
            )?;
        }
        writeln!(f, ")")?;
        Ok(())
    }
}

impl<T: AsRef<[S]>, S: StateID> Automaton for DFA<T, S> {
    type ID = S;

    #[inline]
    fn match_offset(&self) -> usize {
        MATCH_OFFSET
    }

    #[inline]
    fn start_state_forward(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_fwd(bytes, start, end);
        self.starts()[index.as_usize()]
    }

    #[inline]
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_rev(bytes, start, end);
        self.starts()[index.as_usize()]
    }

    #[inline]
    fn is_special_state(&self, id: S) -> bool {
        self.special.is_special_state(id)
    }

    #[inline]
    fn is_dead_state(&self, id: S) -> bool {
        self.special.is_dead_state(id)
    }

    #[inline]
    fn is_quit_state(&self, id: S) -> bool {
        self.special.is_quit_state(id)
    }

    #[inline]
    fn is_match_state(&self, id: S) -> bool {
        self.special.is_match_state(id)
    }

    #[inline]
    fn next_state(&self, current: S, input: u8) -> S {
        let input = self.byte_classes().get(input);
        let o = current.as_usize() + input as usize;
        self.trans()[o]
    }

    #[inline]
    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        let input = self.byte_classes().get_unchecked(input);
        let o = current.as_usize() + input as usize;
        *self.trans().get_unchecked(o)
    }

    #[inline]
    fn next_eof_state(&self, current: S) -> S {
        let eof = self.byte_classes().eof().as_usize();
        let o = current.as_usize() + eof;
        self.trans()[o]
    }
}

/// An iterator over all states in a DFA.
///
/// This iterator yields a tuple for each state. The first element of the
/// tuple corresponds to a state's identifier, and the second element
/// corresponds to the state itself (comprised of its transitions).
///
/// `'a` corresponding to the lifetime of original DFA, `T` corresponds to
/// the type of the transition table itself and `S` corresponds to the state
/// identifier representation.
#[cfg(feature = "std")]
pub(crate) struct StateIter<'a, T, S> {
    dfa: &'a DFA<T, S>,
    it: iter::Enumerate<slice::Chunks<'a, S>>,
}

#[cfg(feature = "std")]
impl<'a, T: AsRef<[S]>, S: StateID> Iterator for StateIter<'a, T, S> {
    type Item = (S, State<'a, S>);

    fn next(&mut self) -> Option<(S, State<'a, S>)> {
        self.it.next().map(|(index, chunk)| {
            let id = self.dfa.from_index(index);
            let state = State { transitions: chunk };
            (id, state)
        })
    }
}

/// An immutable representation of a single DFA state.
///
/// `'a` correspondings to the lifetime of a DFA's transition table and `S`
/// corresponds to the state identifier representation.
#[cfg(feature = "std")]
pub(crate) struct State<'a, S> {
    transitions: &'a [S],
}

#[cfg(feature = "std")]
impl<'a, S: StateID> State<'a, S> {
    /// Return an iterator over all transitions in this state. This yields
    /// a number of transitions equivalent to the alphabet length of the
    /// corresponding DFA.
    ///
    /// Each transition is represented by a tuple. The first element is
    /// the input byte for that transition and the second element is the
    /// transitions itself.
    pub fn transitions(&self) -> StateTransitionIter<'_, S> {
        StateTransitionIter {
            len: self.transitions.len(),
            it: self.transitions.iter().enumerate(),
        }
    }

    /// Return an iterator over a sparse representation of the transitions in
    /// this state. Only non-dead transitions are returned.
    ///
    /// The "sparse" representation in this case corresponds to a sequence of
    /// triples. The first two elements of the triple comprise an inclusive
    /// byte range while the last element corresponds to the transition taken
    /// for all bytes in the range.
    ///
    /// This is somewhat more condensed than the classical sparse
    /// representation (where you have an element for every non-dead
    /// transition), but in practice, checking if a byte is in a range is very
    /// cheap and using ranges tends to conserve quite a bit more space.
    pub fn sparse_transitions(&self) -> StateSparseTransitionIter<'_, S> {
        StateSparseTransitionIter { dense: self.transitions(), cur: None }
    }

    /// Returns the number of transitions in this state. This also corresponds
    /// to the alphabet length of this DFA.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}

#[cfg(feature = "std")]
impl<'a, S: StateID> fmt::Debug for State<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut transitions = vec![];
        for (start, end, next_id) in self.sparse_transitions() {
            let index = if f.alternate() {
                next_id.as_usize()
            } else {
                next_id.as_usize() / self.len()
            };
            let line = if start == end {
                format!("{} => {}", start.escape(), index)
            } else {
                format!("{}-{} => {}", start.escape(), end.escape(), index)
            };
            transitions.push(line);
        }
        write!(f, "{}", transitions.join(", "))?;
        Ok(())
    }
}

/// A mutable representation of a single DFA state.
///
/// `'a` correspondings to the lifetime of a DFA's transition table and `S`
/// corresponds to the state identifier representation.
#[cfg(feature = "std")]
pub(crate) struct StateMut<'a, S> {
    transitions: &'a mut [S],
}

#[cfg(feature = "std")]
impl<'a, S: StateID> StateMut<'a, S> {
    /// Return an iterator over all transitions in this state. This yields
    /// a number of transitions equivalent to the alphabet length of the
    /// corresponding DFA.
    ///
    /// Each transition is represented by a tuple. The first element is the
    /// input byte for that transition and the second element is a mutable
    /// reference to the transition itself.
    pub fn iter_mut(&mut self) -> StateTransitionIterMut<'_, S> {
        StateTransitionIterMut {
            len: self.transitions.len(),
            it: self.transitions.iter_mut().enumerate(),
        }
    }
}

#[cfg(feature = "std")]
impl<'a, S: StateID> fmt::Debug for StateMut<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&State { transitions: self.transitions }, f)
    }
}

/// An iterator over all transitions in a single DFA state. This yields
/// a number of transitions equivalent to the alphabet length of the
/// corresponding DFA.
///
/// Each transition is represented by a tuple. The first element is the input
/// byte for that transition and the second element is the transitions itself.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct StateTransitionIter<'a, S> {
    len: usize,
    it: iter::Enumerate<slice::Iter<'a, S>>,
}

#[cfg(feature = "std")]
impl<'a, S: StateID> Iterator for StateTransitionIter<'a, S> {
    type Item = (Byte, S);

    fn next(&mut self) -> Option<(Byte, S)> {
        self.it.next().map(|(i, &id)| {
            let b = if i + 1 == self.len {
                Byte::EOF(i as u16)
            } else {
                Byte::U8(i as u8)
            };
            (b, id)
        })
    }
}

/// A mutable iterator over all transitions in a DFA state.
///
/// Each transition is represented by a tuple. The first element is the
/// input byte for that transition and the second element is a mutable
/// reference to the transition itself.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct StateTransitionIterMut<'a, S> {
    len: usize,
    it: iter::Enumerate<slice::IterMut<'a, S>>,
}

#[cfg(feature = "std")]
impl<'a, S: StateID> Iterator for StateTransitionIterMut<'a, S> {
    type Item = (Byte, &'a mut S);

    fn next(&mut self) -> Option<(Byte, &'a mut S)> {
        self.it.next().map(|(i, id)| {
            let b = if i + 1 == self.len {
                Byte::EOF(i as u16)
            } else {
                Byte::U8(i as u8)
            };
            (b, id)
        })
    }
}

/// An iterator over all transitions in a single DFA state using a sparse
/// representation.
///
/// Each transition is represented by a triple. The first two elements of the
/// triple comprise an inclusive byte range while the last element corresponds
/// to the transition taken for all bytes in the range.
///
/// As a convenience, this always returns `Byte` values of the same type. That
/// is, you'll never get a (Byte::U8, Byte::EOF) or a (Byte::EOF, Byte::U8).
/// Only (Byte::U8, Byte::U8) and (Byte::EOF, Byte::EOF) values are yielded.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct StateSparseTransitionIter<'a, S> {
    dense: StateTransitionIter<'a, S>,
    cur: Option<(Byte, Byte, S)>,
}

#[cfg(feature = "std")]
impl<'a, S: StateID> Iterator for StateSparseTransitionIter<'a, S> {
    type Item = (Byte, Byte, S);

    fn next(&mut self) -> Option<(Byte, Byte, S)> {
        while let Some((b, next)) = self.dense.next() {
            let (prev_start, prev_end, prev_next) = match self.cur {
                Some(t) => t,
                None => {
                    self.cur = Some((b, b, next));
                    continue;
                }
            };
            if prev_next == next && !b.is_eof() {
                self.cur = Some((prev_start, b, prev_next));
            } else {
                self.cur = Some((b, b, next));
                if prev_next != dead_id() {
                    return Some((prev_start, prev_end, prev_next));
                }
            }
        }
        if let Some((start, end, next)) = self.cur.take() {
            if next != dead_id() {
                return Some((start, end, next));
            }
        }
        None
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn tiny_dfa_works() {
        let pattern = r"\w";
        Builder::new()
            .configure(Config::new().anchored(true))
            .syntax(crate::SyntaxConfig::new().unicode(false))
            .build_with_size::<u8>(pattern)
            .unwrap();
    }

    #[test]
    fn errors_when_converting_to_smaller_dfa() {
        let pattern = r"\w{10}";
        let dfa = Builder::new()
            .configure(Config::new().anchored(true).byte_classes(false))
            .build_with_size::<u32>(pattern)
            .unwrap();
        assert!(dfa.to_u16().is_err());
    }

    #[test]
    fn errors_when_determinization_would_overflow() {
        let pattern = r"\w{10}";

        let mut builder = Builder::new();
        builder.configure(Config::new().anchored(true).byte_classes(false));
        // using u32 is fine
        assert!(builder.build_with_size::<u32>(pattern).is_ok());
        // // ... but u16 results in overflow (because there are >65536 states)
        assert!(builder.build_with_size::<u16>(pattern).is_err());
    }

    #[test]
    fn errors_when_classes_would_overflow() {
        let pattern = r"[a-z]";

        let mut builder = Builder::new();
        builder.configure(Config::new().anchored(true).byte_classes(true));
        // with classes is OK
        assert!(builder.build_with_size::<u8>(pattern).is_ok());
        // ... but without classes, it fails, since states become much bigger.
        builder.configure(Config::new().byte_classes(false));
        assert!(builder.build_with_size::<u8>(pattern).is_err());
    }

    #[test]
    fn errors_with_unicode_word_boundary() {
        let pattern = r"\b";
        assert!(Builder::new().build(pattern).is_err());
    }
}
