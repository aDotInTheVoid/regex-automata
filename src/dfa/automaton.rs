use crate::dfa::search;
use crate::state_id::StateID;
use crate::word::is_word_byte;
use crate::NoMatch;

/// The size of the alphabet in a standard DFA.
///
/// Specifically, this length controls the number of transitions present in
/// each DFA state. However, when the byte class optimization is enabled,
/// then each DFA maps the space of all possible 256 byte values to at most
/// 256 distinct equivalence classes. In this case, the number of distinct
/// equivalence classes corresponds to the internal alphabet of the DFA, in the
/// sense that each DFA state has a number of transitions equal to the number
/// of equivalence classes despite supporting matching on all possible byte
/// values.
pub const ALPHABET_LEN: usize = 256 + 1;

/// The offset, in bytes, that a match is delayed by in the DFAs generated
/// by this crate.
pub const MATCH_OFFSET: usize = 1;

/// The special EOF sentinel value.
pub const EOF: usize = ALPHABET_LEN - 1;

/// A trait describing the interface of a deterministic finite automaton (DFA).
///
/// Every DFA has exactly one start state and at least one dead state (which
/// may be the same, as in the case of an empty DFA). In all cases, a state
/// identifier of `0` must be a dead state such that `DFA::is_dead_state(0)`
/// always returns `true`.
///
/// Every DFA also has zero or more match states, such that
/// `DFA::is_match_state(id)` returns `true` if and only if `id` corresponds to
/// a match state.
///
/// In general, users of this trait likely will only need to use the search
/// routines such as `is_match`, `shortest_match`, `find` or `rfind`. The other
/// methods are lower level and are used for walking the transitions of a DFA
/// manually. In particular, the aforementioned search routines are implemented
/// generically in terms of the lower level transition walking routines.
pub trait Automaton: core::fmt::Debug {
    /// The representation used for state identifiers in this DFA.
    ///
    /// Typically, this is one of `u8`, `u16`, `u32`, `u64` or `usize`.
    type ID: StateID;

    /// Return the match offset of this DFA. This corresponds to the number
    /// of bytes that a match is delayed by. This is typically set to `1`,
    /// which means that a match is always reported exactly one byte after it
    /// occurred.
    fn match_offset(&self) -> usize;

    /// Return the identifier of this DFA's start state for the given haystack
    /// when matching in the forward direction.
    fn start_state_forward(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID;

    /// Return the identifier of this DFA's start state for the given haystack
    /// when matching in the reverse direction.
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID;

    /// Returns true if and only if the given identifier corresponds to either
    /// a dead state or a match state, such that one of `is_match_state(id)`
    /// or `is_dead_state(id)` must return true.
    ///
    /// Depending on the implementation of the DFA, this routine can be used
    /// to save a branch in the core matching loop. Nevertheless,
    /// `is_match_state(id) || is_dead_state(id)` is always a valid
    /// implementation.
    fn is_special_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a dead
    /// state. When a DFA enters a dead state, it is impossible to leave and
    /// thus can never lead to a match.
    fn is_dead_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a
    /// quit state. A quit state is like a dead state (it has no outgoing
    /// transitions), except it indicates that the DFA failed to complete the
    /// search. When this occurs, callers can neither accept or reject that a
    /// match occurred.
    fn is_quit_state(&self, id: Self::ID) -> bool;

    /// Returns true if and only if the given identifier corresponds to a match
    /// state.
    fn is_match_state(&self, id: Self::ID) -> bool;

    /// Given the current state that this DFA is in and the next input byte,
    /// this method returns the identifier of the next state. The identifier
    /// returned is always valid, but it may correspond to a dead state.
    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID;

    /// Like `next_state`, but its implementation may look up the next state
    /// without memory safety checks such as bounds checks. As such, callers
    /// must ensure that the given identifier corresponds to a valid DFA
    /// state. Implementors must, in turn, ensure that this routine is safe
    /// for all valid state identifiers and for all possible `u8` values.
    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID;

    /// Given the current state and an input that has reached EOF, attempt the
    /// final state transition.
    ///
    /// For DFAs that do not delay matches, this should always return the given
    /// state ID.
    fn next_eof_state(&self, current: Self::ID) -> Self::ID;

    /// Returns the first position at which a match is found.
    ///
    /// This routine stops scanning input in precisely the same circumstances
    /// as `is_match`. The key difference is that this routine returns the
    /// position at which it stopped scanning input if and only if a match
    /// was found. If no match is found, then `None` is returned.
    ///
    /// # Example
    ///
    /// This example shows how to use this method with a
    /// [`dense::DFA`](struct.DFA.html).
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa = dense::DFA::new("foo[0-9]+")?;
    /// assert_eq!(Ok(Some(4)), dfa.find_earliest_fwd(b"foo12345"));
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let dfa = dense::DFA::new("abc|a")?;
    /// assert_eq!(Ok(Some(1)), dfa.find_earliest_fwd(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    #[inline]
    fn find_earliest_fwd(
        &self,
        bytes: &[u8],
    ) -> Result<Option<usize>, NoMatch> {
        self.find_earliest_fwd_at(bytes, 0, bytes.len())
    }

    #[inline]
    fn find_earliest_rev(
        &self,
        bytes: &[u8],
    ) -> Result<Option<usize>, NoMatch> {
        self.find_earliest_rev_at(bytes, 0, bytes.len())
    }

    /// Returns the end offset of the longest match. If no match exists,
    /// then `None` is returned.
    ///
    /// Implementors of this trait are not required to implement any particular
    /// match semantics (such as leftmost-first), which are instead manifest in
    /// the DFA's topology itself.
    ///
    /// In particular, this method must continue searching even after it
    /// enters a match state. The search should only terminate once it has
    /// reached the end of the input or when it has entered a dead state. Upon
    /// termination, the position of the last byte seen while still in a match
    /// state is returned.
    ///
    /// # Example
    ///
    /// This example shows how to use this method with a
    /// [`dense::DFA`](struct.DFA.html). By default, a dense DFA uses
    /// "leftmost first" match semantics.
    ///
    /// Leftmost first match semantics corresponds to the match with the
    /// smallest starting offset, but where the end offset is determined by
    /// preferring earlier branches in the original regular expression. For
    /// example, `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam`
    /// will match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, dense};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa = dense::DFA::new("foo[0-9]+")?;
    /// assert_eq!(Ok(Some(8)), dfa.find_leftmost_fwd(b"foo12345"));
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let dfa = dense::DFA::new("abc|a")?;
    /// assert_eq!(Ok(Some(3)), dfa.find_leftmost_fwd(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    #[inline]
    fn find_leftmost_fwd(
        &self,
        bytes: &[u8],
    ) -> Result<Option<usize>, NoMatch> {
        self.find_leftmost_fwd_at(bytes, 0, bytes.len())
    }

    /// Returns the start offset of the longest match in reverse, by searching
    /// from the end of the input towards the start of the input. If no match
    /// exists, then `None` is returned. In other words, this has the same
    /// match semantics as `find`, but in reverse.
    ///
    /// # Example
    ///
    /// This example shows how to use this method with a
    /// [`dense::DFA`](struct.DFA.html). In particular, this routine
    /// is principally useful when used in conjunction with the
    /// [`dense::Builder::reverse`](dense/struct.Builder.html#method.reverse)
    /// configuration knob. In general, it's unlikely to be correct to use both
    /// `find` and `rfind` with the same DFA since any particular DFA will only
    /// support searching in one direction.
    ///
    /// ```
    /// use regex_automata::nfa::thompson;
    /// use regex_automata::dfa::{dense, Automaton};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa = dense::Builder::new()
    ///     .thompson(thompson::Config::new().reverse(true))
    ///     .build("foo[0-9]+")?;
    /// assert_eq!(Ok(Some(0)), dfa.find_leftmost_rev(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    #[inline]
    fn find_leftmost_rev(
        &self,
        bytes: &[u8],
    ) -> Result<Option<usize>, NoMatch> {
        self.find_leftmost_rev_at(bytes, 0, bytes.len())
    }

    #[inline]
    fn find_overlapping_fwd(
        &self,
        bytes: &[u8],
        state_id: &mut Option<Self::ID>,
    ) -> Result<Option<usize>, NoMatch> {
        self.find_overlapping_fwd_at(bytes, 0, bytes.len(), state_id)
    }

    /// Returns the same as `shortest_match`, but starts the search at the
    /// given offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    #[inline]
    fn find_earliest_fwd_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<usize>, NoMatch> {
        search::find_earliest_fwd(self, bytes, start, end)
    }

    #[inline]
    fn find_earliest_rev_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<usize>, NoMatch> {
        search::find_earliest_rev(self, bytes, start, end)
    }

    /// Returns the same as `find`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    #[inline]
    fn find_leftmost_fwd_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<usize>, NoMatch> {
        search::find_leftmost_fwd(self, bytes, start, end)
    }

    /// Returns the same as `rfind`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == bytes.len()`.
    #[inline]
    fn find_leftmost_rev_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<usize>, NoMatch> {
        search::find_leftmost_rev(self, bytes, start, end)
    }

    #[inline]
    fn find_overlapping_fwd_at(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
        state_id: &mut Option<Self::ID>,
    ) -> Result<Option<usize>, NoMatch> {
        search::find_overlapping_fwd(self, bytes, start, end, state_id)
    }
}

impl<'a, T: Automaton> Automaton for &'a T {
    type ID = T::ID;

    #[inline]
    fn match_offset(&self) -> usize {
        (**self).match_offset()
    }

    #[inline]
    fn start_state_forward(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID {
        (**self).start_state_forward(bytes, start, end)
    }

    #[inline]
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Self::ID {
        (**self).start_state_reverse(bytes, start, end)
    }

    #[inline]
    fn is_match_state(&self, id: Self::ID) -> bool {
        (**self).is_match_state(id)
    }

    #[inline]
    fn is_special_state(&self, id: Self::ID) -> bool {
        (**self).is_special_state(id)
    }

    #[inline]
    fn is_dead_state(&self, id: Self::ID) -> bool {
        (**self).is_dead_state(id)
    }

    #[inline]
    fn is_quit_state(&self, id: Self::ID) -> bool {
        (**self).is_quit_state(id)
    }

    #[inline]
    fn next_state(&self, current: Self::ID, input: u8) -> Self::ID {
        (**self).next_state(current, input)
    }

    #[inline]
    unsafe fn next_state_unchecked(
        &self,
        current: Self::ID,
        input: u8,
    ) -> Self::ID {
        (**self).next_state_unchecked(current, input)
    }

    fn next_eof_state(&self, current: Self::ID) -> Self::ID {
        (**self).next_eof_state(current)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Start {
    AfterNonWordByte = 0,
    AfterWordByte = 1,
    Text = 2,
    Line = 3,
}

impl Start {
    pub fn all() -> [Start; 4] {
        [
            Start::AfterNonWordByte,
            Start::AfterWordByte,
            Start::Text,
            Start::Line,
        ]
    }

    pub fn from_usize(n: usize) -> Option<Start> {
        match n {
            0 => Some(Start::AfterNonWordByte),
            1 => Some(Start::AfterWordByte),
            2 => Some(Start::Text),
            3 => Some(Start::Line),
            _ => None,
        }
    }

    pub fn count() -> usize {
        4
    }

    pub fn from_position_fwd(bytes: &[u8], start: usize, end: usize) -> Start {
        if start == 0 {
            Start::Text
        } else if bytes[start - 1] == b'\n' {
            Start::Line
        } else if is_word_byte(bytes[start - 1]) {
            Start::AfterWordByte
        } else {
            Start::AfterNonWordByte
        }
    }

    pub fn from_position_rev(bytes: &[u8], start: usize, end: usize) -> Start {
        if end == bytes.len() {
            Start::Text
        } else if bytes[end] == b'\n' {
            Start::Line
        } else if is_word_byte(bytes[end]) {
            Start::AfterWordByte
        } else {
            Start::AfterNonWordByte
        }
    }

    pub fn as_usize(&self) -> usize {
        *self as usize
    }
}
