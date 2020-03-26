use crate::dfa::automaton::Automaton;
#[cfg(feature = "std")]
use crate::dfa::dense;
#[cfg(feature = "std")]
use crate::dfa::error::Error;
#[cfg(feature = "std")]
use crate::dfa::sparse;
use crate::nfa::thompson;
#[cfg(feature = "std")]
use crate::state_id::StateID;
use crate::{Match, MatchKind, NoMatch};

/// A regular expression that uses deterministic finite automata for fast
/// searching.
///
/// A regular expression is comprised of two DFAs, a "forward" DFA and a
/// "reverse" DFA. The forward DFA is responsible for detecting the end of a
/// match while the reverse DFA is responsible for detecting the start of a
/// match. Thus, in order to find the bounds of any given match, a forward
/// search must first be run followed by a reverse search. A match found by
/// the forward DFA guarantees that the reverse DFA will also find a match.
///
/// The type of the DFA used by a `Regex` corresponds to the `A` type
/// parameter, which must satisfy the [`Automaton`](trait.Automaton.html)
/// trait. Typically, `A` is either a
/// [`dense::DFA`](dense/struct.DFA.html)
/// or a
/// [`sparse::DFA`](sparse/struct.DFA.html),
/// where dense DFAs use more memory but search faster, while sparse DFAs use
/// less memory but search more slowly.
///
/// By default, a regex's automaton type parameter is set to
/// `dense::DFA<Vec<usize>, usize>`. For most in-memory work loads, this is the
/// most convenient type that gives the best search performance.
///
/// # Sparse DFAs
///
/// Since a `Regex` is generic over the `Automaton` trait, it can be used with
/// any kind of DFA. While this crate constructs dense DFAs by default, it is
/// easy enough to build corresponding sparse DFAs, and then build a regex from
/// them:
///
/// ```
/// use regex_automata::dfa::Regex;
///
/// # fn example() -> Result<(), regex_automata::dfa::Error> {
/// // First, build a regex that uses dense DFAs.
/// let dense_re = Regex::new("foo[0-9]+")?;
///
/// // Second, build sparse DFAs from the forward and reverse dense DFAs.
/// let fwd = dense_re.forward().to_sparse()?;
/// let rev = dense_re.reverse().to_sparse()?;
///
/// // Third, build a new regex from the constituent sparse DFAs.
/// let sparse_re = Regex::from_dfas(fwd, rev);
///
/// // A regex that uses sparse DFAs can be used just like with dense DFAs.
/// assert_eq!(true, sparse_re.is_match(b"foo123"));
/// # Ok(()) }; example().unwrap()
/// ```
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct Regex<A = dense::DFA<Vec<usize>, usize>> {
    forward: A,
    reverse: A,
}

/// A regular expression that uses deterministic finite automata for fast
/// searching.
///
/// A regular expression is comprised of two DFAs, a "forward" DFA and a
/// "reverse" DFA. The forward DFA is responsible for detecting the end of a
/// match while the reverse DFA is responsible for detecting the start of a
/// match. Thus, in order to find the bounds of any given match, a forward
/// search must first be run followed by a reverse search. A match found by
/// the forward DFA guarantees that the reverse DFA will also find a match.
///
/// The type of the DFA used by a `Regex` corresponds to the `A` type
/// parameter, which must satisfy the [`Automaton`](trait.Automaton.html)
/// trait. Typically, `A` is either a
/// [`dense::DFA`](dense/struct.DFA.html)
/// or a
/// [`sparse::DFA`](sparse/struct.DFA.html),
/// where dense DFAs use more memory but search faster, while sparse DFAs use
/// less memory but search more slowly.
///
/// When using this crate without the standard library, the `Regex` type has
/// no default type parameter.
///
/// # Sparse DFAs
///
/// Since a `Regex` is generic over the `Automaton` trait, it can be used with
/// any kind of DFA. While this crate constructs dense DFAs by default, it is
/// easy enough to build corresponding sparse DFAs, and then build a regex from
/// them:
///
/// ```
/// use regex_automata::dfa::Regex;
///
/// # fn example() -> Result<(), regex_automata::dfa::Error> {
/// // First, build a regex that uses dense DFAs.
/// let dense_re = Regex::new("foo[0-9]+")?;
///
/// // Second, build sparse DFAs from the forward and reverse dense DFAs.
/// let fwd = dense_re.forward().to_sparse()?;
/// let rev = dense_re.reverse().to_sparse()?;
///
/// // Third, build a new regex from the constituent sparse DFAs.
/// let sparse_re = Regex::from_dfas(fwd, rev);
///
/// // A regex that uses sparse DFAs can be used just like with dense DFAs.
/// assert_eq!(true, sparse_re.is_match(b"foo123"));
/// # Ok(()) }; example().unwrap()
/// ```
#[cfg(not(feature = "std"))]
#[derive(Clone, Debug)]
pub struct Regex<A> {
    forward: A,
    reverse: A,
}

#[cfg(feature = "std")]
impl Regex {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding regex.
    ///
    /// The default configuration uses `usize` for state IDs. The underlying
    /// DFAs are *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`RegexBuilder`](struct.RegexBuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::Regex};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(
    ///     Some(Match::new(3, 14)),
    ///     re.find_leftmost(b"zzzfoo12345barzzz"),
    /// );
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new(pattern: &str) -> Result<Regex, Error> {
        RegexBuilder::new().build(pattern)
    }
}

#[cfg(feature = "std")]
impl Regex<sparse::DFA<Vec<u8>, usize>> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding regex using sparse DFAs.
    ///
    /// The default configuration uses `usize` for state IDs, reduces the
    /// alphabet size by splitting bytes into equivalence classes. The
    /// underlying DFAs are *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`RegexBuilder`](struct.RegexBuilder.html)
    /// to set your own configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::Regex};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let re = Regex::new_sparse("foo[0-9]+bar")?;
    /// assert_eq!(
    ///     Some(Match::new(3, 14)),
    ///     re.find_leftmost(b"zzzfoo12345barzzz"),
    /// );
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new_sparse(
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>, usize>>, Error> {
        RegexBuilder::new().build_sparse(pattern)
    }
}

impl<A: Automaton> Regex<A> {
    /// Returns true if and only if the given bytes match.
    ///
    /// This routine may short circuit if it knows that scanning future input
    /// will never lead to a different result. In particular, if the underlying
    /// DFA enters a match state or a dead state, then this routine will return
    /// `true` or `false`, respectively, without inspecting any future input.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let re = Regex::new("foo[0-9]+bar")?;
    /// assert_eq!(true, re.is_match(b"foo12345bar"));
    /// assert_eq!(false, re.is_match(b"foobar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn is_match(&self, input: &[u8]) -> bool {
        self.is_match_at(input, 0, input.len())
    }

    /// Returns the first position at which a match is found.
    ///
    /// This routine stops scanning input in precisely the same circumstances
    /// as `is_match`. The key difference is that this routine returns the
    /// position at which it stopped scanning input if and only if a match
    /// was found. If no match is found, then `None` is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::Regex};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(Some(Match::new(0, 4)), re.find_earliest(b"foo12345"));
    ///
    /// // Normally, the end of the leftmost first match here would be 3,
    /// // but the shortest match semantics detect a match earlier.
    /// let re = Regex::new("abc|a")?;
    /// assert_eq!(Some(Match::new(0, 1)), re.find_earliest(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn find_earliest(&self, input: &[u8]) -> Option<Match> {
        self.find_earliest_at(input, 0, input.len())
    }

    /// Returns the start and end offset of the leftmost first match. If no
    /// match exists, then `None` is returned.
    ///
    /// The "leftmost first" match corresponds to the match with the smallest
    /// starting offset, but where the end offset is determined by preferring
    /// earlier branches in the original regular expression. For example,
    /// `Sam|Samwise` will match `Sam` in `Samwise`, but `Samwise|Sam` will
    /// match `Samwise` in `Samwise`.
    ///
    /// Generally speaking, the "leftmost first" match is how most backtracking
    /// regular expressions tend to work. This is in contrast to POSIX-style
    /// regular expressions that yield "leftmost longest" matches. Namely,
    /// both `Sam|Samwise` and `Samwise|Sam` match `Samwise` when using
    /// leftmost longest semantics.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::Regex};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(
    ///     Some(Match::new(3, 11)),
    ///     re.find_leftmost(b"zzzfoo12345zzz"),
    /// );
    ///
    /// // Even though a match is found after reading the first byte (`a`),
    /// // the leftmost first match semantics demand that we find the earliest
    /// // match that prefers earlier parts of the pattern over latter parts.
    /// let re = Regex::new("abc|a")?;
    /// assert_eq!(Some(Match::new(0, 3)), re.find_leftmost(b"abc"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn find_leftmost(&self, input: &[u8]) -> Option<Match> {
        self.find_leftmost_at(input, 0, input.len())
    }

    pub fn find_overlapping(
        &self,
        input: &[u8],
        state_id: &mut Option<A::ID>,
    ) -> Option<Match> {
        self.find_overlapping_at(input, 0, input.len(), state_id)
    }

    pub fn find_earliest_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindEarliestMatches<'r, 't, A> {
        FindEarliestMatches::new(self, input)
    }

    /// Returns an iterator over all non-overlapping leftmost first matches
    /// in the given bytes. If no match exists, then the iterator yields no
    /// elements.
    ///
    /// Note that if the regex can match the empty string, then it is
    /// possible for the iterator to yield a zero-width match at a location
    /// that is not a valid UTF-8 boundary (for example, between the code units
    /// of a UTF-8 encoded codepoint). This can happen regardless of whether
    /// [`allow_invalid_utf8`](struct.RegexBuilder.html#method.allow_invalid_utf8)
    /// was enabled or not.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::{Match, dfa::Regex};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let re = Regex::new("foo[0-9]+")?;
    /// let text = b"foo1 foo12 foo123";
    /// let matches: Vec<Match> = re.find_leftmost_iter(text).collect();
    /// assert_eq!(matches, vec![
    ///     Match::new(0, 4),
    ///     Match::new(5, 10),
    ///     Match::new(11, 17),
    /// ]);
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn find_leftmost_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindLeftmostMatches<'r, 't, A> {
        FindLeftmostMatches::new(self, input)
    }

    pub fn find_overlapping_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindOverlappingMatches<'r, 't, A> {
        FindOverlappingMatches::new(self, input)
    }

    /// Build a new regex from its constituent forward and reverse DFAs.
    ///
    /// This is useful when deserializing a regex from some arbitrary
    /// memory region. This is also useful for building regexes from other
    /// types of DFAs.
    ///
    /// # Example
    ///
    /// This example is a bit a contrived. The usual use of these methods
    /// would involve serializing `initial_re` somewhere and then deserializing
    /// it later to build a regex.
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let (fwd, rev) = (initial_re.forward(), initial_re.reverse());
    /// let re = Regex::from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    ///
    /// This example shows how you might build smaller DFAs, and then use those
    /// smaller DFAs to build a new regex.
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let fwd = initial_re.forward().to_u16()?;
    /// let rev = initial_re.reverse().to_u16()?;
    /// let re = Regex::from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    ///
    /// This example shows how to build a `Regex` that uses sparse DFAs instead
    /// of dense DFAs:
    ///
    /// ```
    /// use regex_automata::dfa::Regex;
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let initial_re = Regex::new("foo[0-9]+")?;
    /// assert_eq!(true, initial_re.is_match(b"foo123"));
    ///
    /// let fwd = initial_re.forward().to_sparse()?;
    /// let rev = initial_re.reverse().to_sparse()?;
    /// let re = Regex::from_dfas(fwd, rev);
    /// assert_eq!(true, re.is_match(b"foo123"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn from_dfas(forward: A, reverse: A) -> Regex<A> {
        Regex { forward, reverse }
    }

    /// Return the underlying DFA responsible for forward matching.
    pub fn forward(&self) -> &A {
        &self.forward
    }

    /// Return the underlying DFA responsible for reverse matching.
    pub fn reverse(&self) -> &A {
        &self.reverse
    }
}

/// Lower level infallible search routines that permit controlling where the
/// search starts and ends in a particular sequence.
impl<A: Automaton> Regex<A> {
    /// Returns the same as `is_match`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn is_match_at(&self, input: &[u8], start: usize, end: usize) -> bool {
        self.try_is_match_at(input, start, end).unwrap()
    }

    /// Returns the same as `earliest_match`, but starts the search at the
    /// given offsets.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn find_earliest_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Option<Match> {
        self.try_find_earliest_at(input, start, end).unwrap()
    }

    /// Returns the same as `find`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn find_leftmost_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Option<Match> {
        self.try_find_leftmost_at(input, start, end).unwrap()
    }

    pub fn find_overlapping_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
        state_id: &mut Option<A::ID>,
    ) -> Option<Match> {
        self.try_find_overlapping_at(input, start, end, state_id).unwrap()
    }
}

/// Fallible search routines. These may return an error when the underlying
/// DFAs have been configured in a way that permits them to fail during a
/// search.
///
/// Errors during search only occur when the DFA has been explicitly
/// configured to do so, usually by specifying one or more "quit" bytes or by
/// heuristically enabling Unicode word boundaries.
///
/// Errors will never be returned using the default configuration. So these
/// fallible routines are only needed for particular configurations.
impl<A: Automaton> Regex<A> {
    pub fn try_is_match(&self, input: &[u8]) -> Result<bool, NoMatch> {
        self.try_is_match_at(input, 0, input.len())
    }

    pub fn try_find_earliest(
        &self,
        input: &[u8],
    ) -> Result<Option<Match>, NoMatch> {
        self.try_find_earliest_at(input, 0, input.len())
    }

    pub fn try_find_leftmost(
        &self,
        input: &[u8],
    ) -> Result<Option<Match>, NoMatch> {
        self.try_find_leftmost_at(input, 0, input.len())
    }

    pub fn try_find_overlapping(
        &self,
        input: &[u8],
        state_id: &mut Option<A::ID>,
    ) -> Result<Option<Match>, NoMatch> {
        self.try_find_overlapping_at(input, 0, input.len(), state_id)
    }

    pub fn try_find_earliest_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindEarliestMatches<'r, 't, A> {
        FindEarliestMatches::new(self, input)
    }

    pub fn try_find_leftmost_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindLeftmostMatches<'r, 't, A> {
        FindLeftmostMatches::new(self, input)
    }

    pub fn try_find_overlapping_iter<'r, 't>(
        &'r self,
        input: &'t [u8],
    ) -> FindOverlappingMatches<'r, 't, A> {
        FindOverlappingMatches::new(self, input)
    }
}

/// Lower level fallible search routines that permit controlling where the
/// search starts and ends in a particular sequence.
impl<A: Automaton> Regex<A> {
    /// Returns the same as `is_match`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn try_is_match_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<bool, NoMatch> {
        self.forward()
            .find_earliest_fwd_at(input, start, end)
            .map(|x| x.is_some())
    }

    /// Returns the same as `earliest_match`, but starts the search at the
    /// given offsets.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn try_find_earliest_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<Match>, NoMatch> {
        let end =
            match self.forward().find_earliest_fwd_at(input, start, end)? {
                None => return Ok(None),
                Some(end) => end,
            };
        let start = self
            .reverse()
            .find_earliest_rev_at(input, start, end)?
            .expect("reverse search must match if forward search does");
        assert!(start <= end);
        Ok(Some(Match::new(start, end)))
    }

    /// Returns the same as `find`, but starts the search at the given
    /// offset.
    ///
    /// The significance of the starting point is that it takes the surrounding
    /// context into consideration. For example, if the DFA is anchored, then
    /// a match can only occur when `start == 0`.
    pub fn try_find_leftmost_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
    ) -> Result<Option<Match>, NoMatch> {
        let end =
            match self.forward().find_leftmost_fwd_at(input, start, end)? {
                None => return Ok(None),
                Some(end) => end,
            };
        let start = self
            .reverse()
            .find_leftmost_rev_at(input, start, end)?
            .expect("reverse search must match if forward search does");
        assert!(start <= end);
        Ok(Some(Match::new(start, end)))
    }

    pub fn try_find_overlapping_at(
        &self,
        input: &[u8],
        start: usize,
        end: usize,
        state_id: &mut Option<A::ID>,
    ) -> Result<Option<Match>, NoMatch> {
        let end = match self
            .forward()
            .find_overlapping_fwd_at(input, start, end, state_id)?
        {
            None => return Ok(None),
            Some(end) => end,
        };
        let start = self
            .reverse()
            .find_leftmost_rev_at(input, 0, end)?
            .expect("reverse search must match if forward search does");
        assert!(start <= end);
        Ok(Some(Match::new(start, end)))
    }
}

/// An iterator over all non-overlapping earliest matches for a particular
/// search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct FindEarliestMatches<'r, 't, A>(TryFindEarliestMatches<'r, 't, A>);

impl<'r, 't, A: Automaton> FindEarliestMatches<'r, 't, A> {
    fn new(
        re: &'r Regex<A>,
        text: &'t [u8],
    ) -> FindEarliestMatches<'r, 't, A> {
        FindEarliestMatches(TryFindEarliestMatches::new(re, text))
    }
}

impl<'r, 't, A: Automaton> Iterator for FindEarliestMatches<'r, 't, A> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct FindLeftmostMatches<'r, 't, A>(TryFindLeftmostMatches<'r, 't, A>);

impl<'r, 't, A: Automaton> FindLeftmostMatches<'r, 't, A> {
    fn new(
        re: &'r Regex<A>,
        text: &'t [u8],
    ) -> FindLeftmostMatches<'r, 't, A> {
        FindLeftmostMatches(TryFindLeftmostMatches::new(re, text))
    }
}

impl<'r, 't, A: Automaton> Iterator for FindLeftmostMatches<'r, 't, A> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct FindOverlappingMatches<'r, 't, A: Automaton>(
    TryFindOverlappingMatches<'r, 't, A>,
);

impl<'r, 't, A: Automaton> FindOverlappingMatches<'r, 't, A> {
    fn new(
        re: &'r Regex<A>,
        text: &'t [u8],
    ) -> FindOverlappingMatches<'r, 't, A> {
        FindOverlappingMatches(TryFindOverlappingMatches::new(re, text))
    }
}

impl<'r, 't, A: Automaton> Iterator for FindOverlappingMatches<'r, 't, A> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        next_unwrap(self.0.next())
    }
}

/// An iterator over all non-overlapping earliest matches for a particular
/// search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct TryFindEarliestMatches<'r, 't, A> {
    re: &'r Regex<A>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 't, A: Automaton> TryFindEarliestMatches<'r, 't, A> {
    fn new(
        re: &'r Regex<A>,
        text: &'t [u8],
    ) -> TryFindEarliestMatches<'r, 't, A> {
        TryFindEarliestMatches { re, text, last_end: 0, last_match: None }
    }
}

impl<'r, 't, A: Automaton> Iterator for TryFindEarliestMatches<'r, 't, A> {
    type Item = Result<Match, NoMatch>;

    fn next(&mut self) -> Option<Result<Match, NoMatch>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_earliest_at(
            self.text,
            self.last_end,
            self.text.len(),
        );
        let m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        if m.is_empty() {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = m.end() + 1;
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(m.end()) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = m.end();
        }
        self.last_match = Some(m.end());
        Some(Ok(m))
    }
}

/// An iterator over all non-overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct TryFindLeftmostMatches<'r, 't, A> {
    re: &'r Regex<A>,
    text: &'t [u8],
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 't, A: Automaton> TryFindLeftmostMatches<'r, 't, A> {
    fn new(
        re: &'r Regex<A>,
        text: &'t [u8],
    ) -> TryFindLeftmostMatches<'r, 't, A> {
        TryFindLeftmostMatches { re, text, last_end: 0, last_match: None }
    }
}

impl<'r, 't, A: Automaton> Iterator for TryFindLeftmostMatches<'r, 't, A> {
    type Item = Result<Match, NoMatch>;

    fn next(&mut self) -> Option<Result<Match, NoMatch>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_leftmost_at(
            self.text,
            self.last_end,
            self.text.len(),
        );
        let m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        if m.is_empty() {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = m.end() + 1;
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(m.end()) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = m.end();
        }
        self.last_match = Some(m.end());
        Some(Ok(m))
    }
}

/// An iterator over all overlapping matches for a particular search.
///
/// The iterator yields a `(usize, usize)` value until no more matches could be
/// found. The first `usize` is the start of the match (inclusive) while the
/// second `usize` is the end of the match (exclusive).
///
/// `A` is the type used to represent the underlying DFAs used by the regex.
/// The lifetime variables are as follows:
///
/// * `'r` is the lifetime of the regular expression value itself.
/// * `'t` is the lifetime of the text being searched.
#[derive(Clone, Debug)]
pub struct TryFindOverlappingMatches<'r, 't, A: Automaton> {
    re: &'r Regex<A>,
    text: &'t [u8],
    last_end: usize,
    state_id: Option<A::ID>,
}

impl<'r, 't, A: Automaton> TryFindOverlappingMatches<'r, 't, A> {
    fn new(
        re: &'r Regex<A>,
        text: &'t [u8],
    ) -> TryFindOverlappingMatches<'r, 't, A> {
        TryFindOverlappingMatches { re, text, last_end: 0, state_id: None }
    }
}

impl<'r, 't, A: Automaton> Iterator for TryFindOverlappingMatches<'r, 't, A> {
    type Item = Result<Match, NoMatch>;

    fn next(&mut self) -> Option<Result<Match, NoMatch>> {
        if self.last_end > self.text.len() {
            return None;
        }
        let result = self.re.try_find_overlapping_at(
            self.text,
            self.last_end,
            self.text.len(),
            &mut self.state_id,
        );
        let m = match result {
            Err(err) => return Some(Err(err)),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };
        // Unlike the non-overlapping case, we're OK with empty matches at this
        // level. In particular, the overlapping search algorithm is itself
        // responsible for ensuring that progress is always made. (The starting
        // position of the search is incremented by 1 whenever a non-None state
        // ID is given.)
        self.last_end = m.end();
        Some(Ok(m))
    }
}

/// A builder for a regex based on deterministic finite automatons.
///
/// This builder permits configuring several aspects of the construction
/// process such as case insensitivity, Unicode support and various options
/// that impact the size of the underlying DFAs. In some cases, options (like
/// performing DFA minimization) can come with a substantial additional cost.
///
/// This builder generally constructs two DFAs, where one is responsible for
/// finding the end of a match and the other is responsible for finding the
/// start of a match. If you only need to detect whether something matched,
/// or only the end of a match, then you should use a
/// [`dense::Builder`](dense/struct.Builder.html)
/// to construct a single DFA, which is cheaper than building two DFAs.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct RegexBuilder {
    dfa: dense::Builder,
}

#[cfg(feature = "std")]
impl RegexBuilder {
    /// Create a new regex builder with the default configuration.
    pub fn new() -> RegexBuilder {
        RegexBuilder { dfa: dense::Builder::new() }
    }

    /// Build a regex from the given pattern.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build(&self, pattern: &str) -> Result<Regex, Error> {
        self.build_with_size::<usize>(pattern)
    }

    /// Build a regex from the given pattern using sparse DFAs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    pub fn build_sparse(
        &self,
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>, usize>>, Error> {
        self.build_with_size_sparse::<usize>(pattern)
    }

    /// Build a regex from the given pattern using a specific representation
    /// for the underlying DFA state IDs.
    ///
    /// If there was a problem parsing or compiling the pattern, then an error
    /// is returned.
    ///
    /// The representation of state IDs is determined by the `S` type
    /// parameter. In general, `S` is usually one of `u8`, `u16`, `u32`, `u64`
    /// or `usize`, where `usize` is the default used for `build`. The purpose
    /// of specifying a representation for state IDs is to reduce the memory
    /// footprint of the underlying DFAs.
    ///
    /// When using this routine, the chosen state ID representation will be
    /// used throughout determinization and minimization, if minimization was
    /// requested. Even if the minimized DFAs can fit into the chosen state ID
    /// representation but the initial determinized DFA cannot, then this will
    /// still return an error. To get a minimized DFA with a smaller state ID
    /// representation, first build it with a bigger state ID representation,
    /// and then shrink the sizes of the DFAs using one of its conversion
    /// routines, such as [`dense::DFA::to_u16`](struct.DFA.html#method.to_u16).
    /// Finally, reconstitute the regex via
    /// [`Regex::from_dfa`](struct.Regex.html#method.from_dfa).
    pub fn build_with_size<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<Regex<dense::DFA<Vec<S>, S>>, Error> {
        let forward = self.dfa.build_with_size(pattern)?;
        let reverse = self
            .dfa
            .clone()
            .configure(
                dense::Config::new().anchored(true).match_kind(MatchKind::All),
            )
            .thompson(thompson::Config::new().reverse(true))
            .build_with_size(pattern)?;
        Ok(Regex::from_dfas(forward, reverse))
    }

    /// Build a regex from the given pattern using a specific representation
    /// for the underlying DFA state IDs using sparse DFAs.
    pub fn build_with_size_sparse<S: StateID>(
        &self,
        pattern: &str,
    ) -> Result<Regex<sparse::DFA<Vec<u8>, S>>, Error> {
        let re = self.build_with_size(pattern)?;
        let fwd = re.forward().to_sparse()?;
        let rev = re.reverse().to_sparse()?;
        Ok(Regex::from_dfas(fwd, rev))
    }

    /// Set the syntax configuration for this builder using
    /// [`SyntaxConfig`](../struct.SyntaxConfig.html).
    ///
    /// This permits setting things like case insensitivity, Unicode and multi
    /// line mode.
    pub fn syntax(
        &mut self,
        config: crate::SyntaxConfig,
    ) -> &mut RegexBuilder {
        self.dfa.syntax(config);
        self
    }

    /// Set the Thompson NFA configuration for this builder using
    /// [`nfa::thompson::Config`](../nfa/thompson/struct.Config.html).
    ///
    /// This permits setting things like whether additional time should be
    /// spent shrinking the size of the NFA.
    pub fn thompson(&mut self, config: thompson::Config) -> &mut RegexBuilder {
        self.dfa.thompson(config);
        self
    }

    /// Set the dense DFA compilation configuration for this builder using
    /// [`dfa::dense::Config`](dense/struct.Config.html).
    ///
    /// This permits setting things like whether the underlying DFAs should
    /// be minimized.
    pub fn dense(&mut self, config: dense::Config) -> &mut RegexBuilder {
        self.dfa.configure(config);
        self
    }
}

#[cfg(feature = "std")]
impl Default for RegexBuilder {
    fn default() -> RegexBuilder {
        RegexBuilder::new()
    }
}

fn next_unwrap(item: Option<Result<Match, NoMatch>>) -> Option<Match> {
    match item {
        None => None,
        Some(Ok(m)) => Some(m),
        Some(Err(err)) => panic!(
            "unexpected regex search error: {}\n\
             to handle search errors, use try_ methods",
            err,
        ),
    }
}
