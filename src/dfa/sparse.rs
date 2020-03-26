#[cfg(feature = "std")]
use core::fmt;
#[cfg(feature = "std")]
use core::iter;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::slice;
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(feature = "std")]
use byteorder::{BigEndian, LittleEndian};
use byteorder::{ByteOrder, NativeEndian};

use crate::classes::{Byte, ByteClasses};
use crate::dfa::automaton::{Automaton, Start, MATCH_OFFSET};
use crate::dfa::dense;
#[cfg(feature = "std")]
use crate::dfa::error::Error;
use crate::dfa::special::Special;
#[cfg(feature = "std")]
use crate::state_id::{dead_id, write_state_id_bytes, StateID};
#[cfg(not(feature = "std"))]
use state_id::{dead_id, quit_id, StateID};

/// A sparse table-based deterministic finite automaton (DFA).
///
/// In contrast to a [dense DFA](../dense/struct.DFA.html), a sparse DFA uses a
/// more space efficient representation for its transition table. Consequently,
/// sparse DFAs can use much less memory than dense DFAs, but this comes at a
/// price. In particular, reading the more space efficient transitions takes
/// more work, and consequently, searching using a sparse DFA is typically
/// slower than a dense DFA.
///
/// A sparse DFA can be built using the default configuration via the
/// [`sparse::DFA::new`](struct.DFA.html#method.new) constructor.
/// Otherwise, one can configure various aspects of a dense DFA via
/// [`dense::Builder`](../dense/struct.Builder.html), and then convert a dense
/// DFA to a sparse DFA using
/// [`dense::DFA::to_sparse`](../dense/struct.DFA.html#method.to_sparse).
///
/// In general, a sparse DFA supports all the same operations as a dense DFA.
///
/// Making the choice between a dense and sparse DFA depends on your specific
/// work load. If you can sacrifice a bit of search time performance, then a
/// sparse DFA might be the best choice. In particular, while sparse DFAs are
/// probably always slower than dense DFAs, you may find that they are easily
/// fast enough for your purposes!
///
/// # State size
///
/// A `sparse::DFA` has two type parameters, `T` and `S`. `T` corresponds to
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
/// The type of the transition table is typically either `Vec<u8>` or `&[u8]`,
/// depending on where the transition table is stored. Note that this is
/// different than a dense DFA, whose transition table is typically
/// `Vec<S>` or `&[S]`. The reason for this is that a sparse DFA always reads
/// its transition table from raw bytes because the table is compactly packed.
///
/// # Variants
///
/// This DFA is defined as a non-exhaustive enumeration of different types of
/// dense DFAs. All of the variants use the same internal representation
/// for the transition table, but they vary in how the transition table is
/// read. A DFA's specific variant depends on the configuration options set via
/// [`dense::Builder`](dense/struct.Builder.html). The default variant is
/// `ByteClass`.
///
/// # The `Automaton` trait
///
/// This type implements the [`Automaton`](../trait.DFA.html) trait, which
/// means it can be used for searching. For example:
///
/// ```
/// use regex_automata::dfa::{Automaton, sparse};
///
/// # fn example() -> Result<(), regex_automata::dfa::Error> {
/// let dfa = sparse::DFA::new("foo[0-9]+")?;
/// assert_eq!(Ok(Some(8)), dfa.find_leftmost_fwd(b"foo12345"));
/// # Ok(()) }; example().unwrap()
/// ```
#[derive(Clone)]
pub struct DFA<T, S = usize> {
    state_count: usize,
    special: Special<S>,
    byte_classes: ByteClasses,
    starts: T,
    trans: T,
}

#[cfg(feature = "std")]
impl DFA<Vec<u8>, usize> {
    /// Parse the given regular expression using a default configuration and
    /// return the corresponding sparse DFA.
    ///
    /// The default configuration uses `usize` for state IDs and reduces the
    /// alphabet size by splitting bytes into equivalence classes. The
    /// resulting DFA is *not* minimized.
    ///
    /// If you want a non-default configuration, then use the
    /// [`dense::Builder`](dense/struct.Builder.html)
    /// to set your own configuration, and then call
    /// [`dense::DFA::to_sparse`](struct.DFA.html#method.to_sparse)
    /// to create a sparse DFA.
    ///
    /// # Example
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, sparse};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa = sparse::DFA::new("foo[0-9]+bar")?;
    /// assert_eq!(Ok(Some(11)), dfa.find_leftmost_fwd(b"foo12345bar"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn new(pattern: &str) -> Result<DFA<Vec<u8>, usize>, Error> {
        dense::Builder::new()
            .build(pattern)
            .and_then(|dense| dense.to_sparse())
    }
}

#[cfg(feature = "std")]
impl<S: StateID> DFA<Vec<u8>, S> {
    /// Create a new DFA that matches every input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that always matches, callers must provide a
    /// type hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, sparse};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa: sparse::DFA<Vec<u8>, usize> = sparse::DFA::always_match()?;
    /// assert_eq!(Ok(Some(0)), dfa.find_leftmost_fwd(b""));
    /// assert_eq!(Ok(Some(0)), dfa.find_leftmost_fwd(b"foo"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn always_match() -> Result<DFA<Vec<u8>, S>, Error> {
        dense::DFA::always_match()?.to_sparse()
    }

    /// Create a new sparse DFA that never matches any input.
    ///
    /// # Example
    ///
    /// In order to build a DFA that never matches, callers must provide a type
    /// hint indicating their choice of state identifier representation.
    ///
    /// ```
    /// use regex_automata::dfa::{Automaton, sparse};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let dfa: sparse::DFA<Vec<u8>, usize> = sparse::DFA::never_match()?;
    /// assert_eq!(Ok(None), dfa.find_leftmost_fwd(b""));
    /// assert_eq!(Ok(None), dfa.find_leftmost_fwd(b"foo"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub fn never_match() -> Result<DFA<Vec<u8>, S>, Error> {
        dense::DFA::never_match()?.to_sparse()
    }

    /// The implementation for constructing a sparse DFA from a dense DFA.
    pub(crate) fn from_dense_sized<T: AsRef<[S]>, A: StateID>(
        dfa: &dense::DFA<T, S>,
    ) -> Result<DFA<Vec<u8>, A>, Error> {
        // In order to build the transition table, we need to be able to write
        // state identifiers for each of the "next" transitions in each state.
        // Our state identifiers correspond to the byte offset in the
        // transition table at which the state is encoded. Therefore, we do not
        // actually know what the state identifiers are until we've allocated
        // exactly as much space as we need for each state. Thus, construction
        // of the transition table happens in two passes.
        //
        // In the first pass, we fill out the shell of each state, which
        // includes the transition count, the input byte ranges and zero-filled
        // space for the transitions. In this first pass, we also build up a
        // map from the state identifier index of the dense DFA to the state
        // identifier in this sparse DFA.
        //
        // In the second pass, we fill in the transitions based on the map
        // built in the first pass.

        let mut trans = Vec::with_capacity(size_of::<A>() * dfa.state_count());
        let mut remap: Vec<A> = vec![dead_id(); dfa.state_count()];
        for (old_id, state) in dfa.states() {
            let pos = trans.len();

            remap[dfa.to_index(old_id)] = usize_to_state_id(pos)?;
            // zero-filled space for the transition count
            trans.push(0);
            trans.push(0);

            let mut trans_count = 0;
            for (b1, b2, _) in state.sparse_transitions() {
                match (b1, b2) {
                    (Byte::U8(b1), Byte::U8(b2)) => {
                        trans_count += 1;
                        trans.push(b1);
                        trans.push(b2);
                    }
                    (Byte::EOF(_), Byte::EOF(_)) => {}
                    (Byte::U8(_), Byte::EOF(_))
                    | (Byte::EOF(_), Byte::U8(_)) => {
                        // can never occur because sparse_transitions never
                        // groups EOF with any other transition.
                        unreachable!()
                    }
                }
            }
            // Add dummy EOF transition. This is never actually read while
            // searching, but having space equivalent to the total number
            // of transitions is convenient. Otherwise, we'd need to track
            // a different number of transitions for the byte ranges as for
            // the 'next' states.
            trans_count += 1;
            trans.push(0);
            trans.push(0);

            // fill in the transition count
            NativeEndian::write_u16(&mut trans[pos..], trans_count);

            // zero-fill the actual transitions
            let zeros = trans_count as usize * size_of::<A>();
            trans.extend(iter::repeat(0).take(zeros));
        }

        let mut new = DFA {
            state_count: dfa.state_count(),
            special: dfa.special().remap(|id| remap[dfa.to_index(id)]),
            byte_classes: dfa.byte_classes().clone(),
            starts: vec![0; size_of::<A>() * dfa.starts().len()],
            trans,
        };
        for (i, &old_start_id) in dfa.starts().iter().enumerate() {
            remap[dfa.to_index(old_start_id)]
                .write_bytes(&mut new.starts[i * size_of::<A>()..]);
        }
        for (old_id, old_state) in dfa.states() {
            let new_id = remap[dfa.to_index(old_id)];
            let mut new_state = new.state_mut(new_id);
            let sparse = old_state.sparse_transitions();
            for (i, (_, _, next)) in sparse.enumerate() {
                let next = remap[dfa.to_index(next)];
                new_state.set_next_at(i, next);
            }
        }
        Ok(new)
    }

    /// Return a convenient mutable representation of the given state.
    fn state_mut<'a>(&'a mut self, id: S) -> StateMut<'a, S> {
        let mut pos = id.as_usize();
        let ntrans = NativeEndian::read_u16(&self.trans[pos..]) as usize;
        pos += 2;

        let size = (ntrans * 2) + (ntrans * size_of::<S>());
        let ranges_and_next = &mut self.trans[pos..pos + size];
        let (input_ranges, next) = ranges_and_next.split_at_mut(ntrans * 2);
        StateMut { _state_id_repr: PhantomData, ntrans, input_ranges, next }
    }
}

impl<T: AsRef<[u8]>, S: StateID> DFA<T, S> {
    /// Cheaply return a borrowed version of this sparse DFA. Specifically, the
    /// DFA returned always uses `&[u8]` for its transition table while keeping
    /// the same state identifier representation.
    pub fn as_ref<'a>(&'a self) -> DFA<&'a [u8], S> {
        DFA {
            state_count: self.state_count,
            special: self.special,
            byte_classes: self.byte_classes.clone(),
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
    pub fn to_owned(&self) -> DFA<Vec<u8>, S> {
        DFA {
            state_count: self.state_count,
            special: self.special,
            byte_classes: self.byte_classes.clone(),
            starts: self.starts().to_vec(),
            trans: self.trans().to_vec(),
        }
    }

    /// Returns the memory usage, in bytes, of this DFA.
    ///
    /// The memory usage is computed based on the number of bytes used to
    /// represent this DFA's transition table. This typically corresponds to
    /// heap memory usage.
    ///
    /// This does **not** include the stack size used up by this DFA. To
    /// compute that, used `std::mem::size_of::<sparse::DFA>()`.
    pub fn memory_usage(&self) -> usize {
        self.starts().len() + self.trans().len()
    }

    /// Return a convenient representation of the given state.
    ///
    /// This is marked as inline because it doesn't seem to get inlined
    /// otherwise, which leads to a fairly significant performance loss (~25%).
    #[inline]
    fn state<'a>(&'a self, id: S) -> State<'a, S> {
        let mut pos = id.as_usize();
        let ntrans = NativeEndian::read_u16(&self.trans()[pos..]) as usize;
        pos += 2;
        let input_ranges = &self.trans()[pos..pos + (ntrans * 2)];
        pos += 2 * ntrans;
        let next = &self.trans()[pos..pos + (ntrans * size_of::<S>())];
        State { _state_id_repr: PhantomData, ntrans, input_ranges, next }
    }

    /// Return an iterator over all of the states in this DFA.
    ///
    /// The iterator returned yields tuples, where the first element is the
    /// state ID and the second element is the state itself.
    #[cfg(feature = "std")]
    fn states<'a>(&'a self) -> StateIter<'a, T, S> {
        StateIter { dfa: self, id: dead_id() }
    }

    fn start_ids<'a>(&'a self) -> StartStateIDIter<'a, T, S> {
        StartStateIDIter { dfa: self, i: 0 }
    }

    fn starts(&self) -> &[u8] {
        self.starts.as_ref()
    }

    fn starts_len(&self) -> usize {
        self.starts().len() / size_of::<S>()
    }

    fn trans(&self) -> &[u8] {
        self.trans.as_ref()
    }
}

/// Routines for converting a sparse DFA to other representations, such as
/// smaller state identifiers or raw bytes suitable for persistent storage.
#[cfg(feature = "std")]
impl<T: AsRef<[u8]>, S: StateID> DFA<T, S> {
    /// Create a new sparse DFA whose match semantics are equivalent to
    /// this DFA, but attempt to use `u8` for the representation of state
    /// identifiers. If `u8` is insufficient to represent all state identifiers
    /// in this DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u8>()`.
    pub fn to_u8(&self) -> Result<DFA<Vec<u8>, u8>, Error> {
        self.to_sized()
    }

    /// Create a new sparse DFA whose match semantics are equivalent to
    /// this DFA, but attempt to use `u16` for the representation of state
    /// identifiers. If `u16` is insufficient to represent all state
    /// identifiers in this DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u16>()`.
    pub fn to_u16(&self) -> Result<DFA<Vec<u8>, u16>, Error> {
        self.to_sized()
    }

    /// Create a new sparse DFA whose match semantics are equivalent to
    /// this DFA, but attempt to use `u32` for the representation of state
    /// identifiers. If `u32` is insufficient to represent all state
    /// identifiers in this DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u32>()`.
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    pub fn to_u32(&self) -> Result<DFA<Vec<u8>, u32>, Error> {
        self.to_sized()
    }

    /// Create a new sparse DFA whose match semantics are equivalent to
    /// this DFA, but attempt to use `u64` for the representation of state
    /// identifiers. If `u64` is insufficient to represent all state
    /// identifiers in this DFA, then this returns an error.
    ///
    /// This is a convenience routine for `to_sized::<u64>()`.
    #[cfg(target_pointer_width = "64")]
    pub fn to_u64(&self) -> Result<DFA<Vec<u8>, u64>, Error> {
        self.to_sized()
    }

    /// Create a new sparse DFA whose match semantics are equivalent to
    /// this DFA, but attempt to use `A` for the representation of state
    /// identifiers. If `A` is insufficient to represent all state identifiers
    /// in this DFA, then this returns an error.
    ///
    /// An alternative way to construct such a DFA is to use
    /// [`dense::DFA::to_sparse_sized`](struct.DFA.html#method.to_sparse_sized).
    /// In general, picking the appropriate size upon initial construction of
    /// a sparse DFA is preferred, since it will do the conversion in one
    /// step instead of two.
    pub fn to_sized<A: StateID>(&self) -> Result<DFA<Vec<u8>, A>, Error> {
        // To build the new DFA, we proceed much like the initial construction
        // of the sparse DFA. Namely, since the state ID size is changing,
        // we don't actually know all of our state IDs until we've allocated
        // all necessary space. So we do one pass that allocates all of the
        // storage we need, and then another pass to fill in the transitions.

        let mut starts = vec![0; size_of::<A>() * self.starts_len()];
        let mut trans = Vec::with_capacity(size_of::<A>() * self.state_count);
        let mut map: HashMap<S, A> = HashMap::with_capacity(self.state_count);
        for (old_id, state) in self.states() {
            let pos = trans.len();
            map.insert(old_id, usize_to_state_id(pos)?);

            let n = state.ntrans;
            let zeros = 2 + (n * 2) + (n * size_of::<A>());
            trans.extend(iter::repeat(0).take(zeros));

            NativeEndian::write_u16(&mut trans[pos..], n as u16);
            let (s, e) = (pos + 2, pos + 2 + (n * 2));
            trans[s..e].copy_from_slice(state.input_ranges);
        }
        for (i, old_start_id) in self.start_ids().enumerate() {
            map[&old_start_id].write_bytes(&mut starts[i * size_of::<A>()..]);
        }

        let mut new = DFA {
            state_count: self.state_count,
            special: self.special.remap(|id| map[&id]),
            byte_classes: self.byte_classes.clone(),
            starts,
            trans,
        };
        for (&old_id, &new_id) in map.iter() {
            let old_state = self.state(old_id);
            let mut new_state = new.state_mut(new_id);
            for i in 0..new_state.ntrans {
                let next = map[&old_state.next_at(i)];
                new_state.set_next_at(i, usize_to_state_id(next.as_usize())?);
            }
        }
        Ok(new)
    }

    /// Serialize a sparse DFA to raw bytes in little endian format.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_little_endian(&self) -> Result<Vec<u8>, Error> {
        self.to_bytes::<LittleEndian>()
    }

    /// Serialize a sparse DFA to raw bytes in big endian format.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_big_endian(&self) -> Result<Vec<u8>, Error> {
        self.to_bytes::<BigEndian>()
    }

    /// Serialize a sparse DFA to raw bytes in native endian format.
    /// Generally, it is better to pick an explicit endianness using either
    /// `to_bytes_little_endian` or `to_bytes_big_endian`. This routine is
    /// useful in tests where the DFA is serialized and deserialized on the
    /// same platform.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    pub fn to_bytes_native_endian(&self) -> Result<Vec<u8>, Error> {
        self.to_bytes::<NativeEndian>()
    }

    /// Serialize a sparse DFA to raw bytes using the provided endianness.
    ///
    /// If the state identifier representation of this DFA has a size different
    /// than 1, 2, 4 or 8 bytes, then this returns an error. All
    /// implementations of `StateID` provided by this crate satisfy this
    /// requirement.
    ///
    /// Unlike dense DFAs, the result is not necessarily aligned since a
    /// sparse DFA's transition table is always read as a sequence of bytes.
    #[cfg(feature = "std")]
    fn to_bytes<A: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let label = b"rust-regex-automata-sparse-dfa\x00";
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
            // For DFA misc options. (Currently unused.)
            + 2
            // For state count.
            + 8
            // For start state count.
            + 8
            // For byte class map.
            + 256
            // For special state ID info.
            + <Special<S>>::num_bytes()
            // For start state IDs.
            + self.starts().len()
            // For transition table.
            + self.trans().len();

        let mut i = 0;
        let mut buf = vec![0; size];

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
        let state_size = size_of::<S>();
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
        A::write_u64(&mut buf[i..], self.starts_len() as u64);
        i += 8;
        // byte class map
        for b in (0..256).map(|b| b as u8) {
            buf[i] = self.byte_classes.get(b);
            i += 1;
        }
        // special state ID info
        self.special.to_bytes::<A>(&mut buf[i..]);
        i += <Special<S>>::num_bytes();
        // start state IDs
        for start_id in self.start_ids() {
            write_state_id_bytes::<A, _>(&mut buf[i..], start_id);
            i += size_of::<S>();
        }
        // transition table
        for (_, state) in self.states() {
            A::write_u16(&mut buf[i..], state.ntrans as u16);
            i += 2;
            buf[i..i + (state.ntrans * 2)].copy_from_slice(state.input_ranges);
            i += state.ntrans * 2;
            for j in 0..state.ntrans {
                write_state_id_bytes::<A, _>(&mut buf[i..], state.next_at(j));
                i += size_of::<S>();
            }
        }

        assert_eq!(size, i, "expected to consume entire buffer");

        Ok(buf)
    }
}

impl<'a, S: StateID> DFA<&'a [u8], S> {
    /// Deserialize a sparse DFA with a specific state identifier
    /// representation.
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
    /// are not a valid serialization of a DFA, or if the endianness of the
    /// serialized bytes is different than the endianness of the machine that
    /// is deserializing the DFA, then this routine will panic. Moreover, it
    /// is possible for this deserialization routine to succeed even if the
    /// given bytes do not represent a valid serialized sparse DFA.
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
    /// use regex_automata::dfa::{Automaton, sparse};
    ///
    /// # fn example() -> Result<(), regex_automata::dfa::Error> {
    /// let sparse = sparse::DFA::new("foo[0-9]+")?;
    /// let bytes = sparse.to_u16()?.to_bytes_native_endian()?;
    ///
    /// let dfa: sparse::DFA<&[u8], u16> = unsafe {
    ///     sparse::DFA::from_bytes_unchecked(&bytes)
    /// };
    ///
    /// assert_eq!(Ok(Some(8)), dfa.find_leftmost_fwd(b"foo12345"));
    /// # Ok(()) }; example().unwrap()
    /// ```
    pub unsafe fn from_bytes_unchecked(mut buf: &'a [u8]) -> DFA<&'a [u8], S> {
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
                 are you trying to load a sparse::DFA serialized with a \
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
        if state_size != size_of::<S>() {
            panic!(
                "state size of sparse::DFA ({}) does not match \
                 requested state size ({})",
                state_size,
                size_of::<S>(),
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

        // Read start state and transition tables.
        let starts = &buf[..start_len * state_size];
        let trans = &buf[start_len * state_size..];
        DFA { state_count, special, byte_classes, starts, trans }
    }
}

impl<T: AsRef<[u8]>, S: StateID> Automaton for DFA<T, S> {
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
        S::read_bytes(&self.starts()[index.as_usize() * size_of::<S>()..])
    }

    #[inline]
    fn start_state_reverse(
        &self,
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> S {
        let index = Start::from_position_rev(bytes, start, end);
        S::read_bytes(&self.starts()[index.as_usize() * size_of::<S>()..])
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
        let input = self.byte_classes.get(input);
        self.state(current).next(input)
    }

    #[inline]
    unsafe fn next_state_unchecked(&self, current: S, input: u8) -> S {
        self.next_state(current, input)
    }

    #[inline]
    fn next_eof_state(&self, current: S) -> S {
        self.state(current).next_eof()
    }
}

#[cfg(feature = "std")]
impl<T: AsRef<[u8]>, S: StateID> fmt::Debug for DFA<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn state_status<T: AsRef<[u8]>, S: StateID>(
            dfa: &DFA<T, S>,
            id: S,
        ) -> &'static str {
            if dfa.is_dead_state(id) {
                "D "
            } else if dfa.is_quit_state(id) {
                "Q "
            } else if dfa.start_ids().any(|sid| sid == id) {
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

        writeln!(f, "sparse::DFA(")?;
        for (id, state) in self.states() {
            let status = state_status(self, id);
            writeln!(f, "{}{:06}: {:?}", status, id.as_usize(), state)?;
        }
        writeln!(f, "")?;
        for (i, start_id) in self.start_ids().enumerate() {
            writeln!(
                f,
                "START({}): {:?} => {:06}",
                i,
                Start::from_usize(i),
                start_id.as_usize(),
            )?;
        }
        writeln!(f, ")")?;
        Ok(())
    }
}

/// An iterator over all state state IDs in a sparse DFA.
struct StartStateIDIter<'a, T, S> {
    dfa: &'a DFA<T, S>,
    i: usize,
}

impl<'a, T: AsRef<[u8]>, S: StateID> Iterator for StartStateIDIter<'a, T, S> {
    type Item = S;

    fn next(&mut self) -> Option<S> {
        let pos = self.i * size_of::<S>();
        if pos >= self.dfa.starts().len() {
            None
        } else {
            self.i += 1;
            Some(S::read_bytes(&self.dfa.starts()[pos..]))
        }
    }
}

impl<'a, T, S: StateID> fmt::Debug for StartStateIDIter<'a, T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StartStateIDIter").field("i", &self.i).finish()
    }
}

/// An iterator over all states in a sparse DFA.
///
/// This iterator yields tuples, where the first element is the state ID and
/// the second element is the state itself.
#[cfg(feature = "std")]
struct StateIter<'a, T, S> {
    dfa: &'a DFA<T, S>,
    id: S,
}

#[cfg(feature = "std")]
impl<'a, T: AsRef<[u8]>, S: StateID> Iterator for StateIter<'a, T, S> {
    type Item = (S, State<'a, S>);

    fn next(&mut self) -> Option<(S, State<'a, S>)> {
        if self.id.as_usize() >= self.dfa.trans().len() {
            return None;
        }
        let id = self.id;
        let state = self.dfa.state(id);
        self.id = S::from_usize(self.id.as_usize() + state.bytes());
        Some((id, state))
    }
}

impl<'a, T, S: StateID> fmt::Debug for StateIter<'a, T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StateIter").field("id", &self.id).finish()
    }
}

/// A representation of a sparse DFA state that can be cheaply materialized
/// from a state identifier.
#[derive(Clone)]
struct State<'a, S> {
    /// The state identifier representation used by the DFA from which this
    /// state was extracted. Since our transition table is compacted in a
    /// &[u8], we don't actually use the state ID type parameter explicitly
    /// anywhere, so we fake it. This prevents callers from using an incorrect
    /// state ID representation to read from this state.
    _state_id_repr: PhantomData<S>,
    /// The number of transitions in this state.
    ntrans: usize,
    /// Pairs of input ranges, where there is one pair for each transition.
    /// Each pair specifies an inclusive start and end byte range for the
    /// corresponding transition.
    input_ranges: &'a [u8],
    /// Transitions to the next state. This slice contains native endian
    /// encoded state identifiers, with `S` as the representation. Thus, there
    /// are `ntrans * size_of::<S>()` bytes in this slice.
    next: &'a [u8],
}

impl<'a, S: StateID> State<'a, S> {
    /// Searches for the next transition given an input byte. If no such
    /// transition could be found, then a dead state is returned.
    fn next(&self, input: u8) -> S {
        // This straight linear search was observed to be much better than
        // binary search on ASCII haystacks, likely because a binary search
        // visits the ASCII case last but a linear search sees it first. A
        // binary search does do a little better on non-ASCII haystacks, but
        // not by much. There might be a better trade off lurking here.
        for i in 0..(self.ntrans - 1) {
            let (start, end) = self.range(i);
            if start <= input && input <= end {
                return self.next_at(i);
            }
            // We could bail early with an extra branch: if input < b1, then
            // we know we'll never find a matching transition. Interestingly,
            // this extra branch seems to not help performance, or will even
            // hurt it. It's likely very dependent on the DFA itself and what
            // is being searched.
        }
        dead_id()
    }

    fn next_eof(&self) -> S {
        self.next_at(self.ntrans - 1)
    }

    /// Returns the inclusive input byte range for the ith transition in this
    /// state.
    fn range(&self, i: usize) -> (u8, u8) {
        (self.input_ranges[i * 2], self.input_ranges[i * 2 + 1])
    }

    /// Returns the next state for the ith transition in this state.
    fn next_at(&self, i: usize) -> S {
        S::read_bytes(&self.next[i * size_of::<S>()..])
    }

    /// Return the total number of bytes that this state consumes in its
    /// encoded form.
    #[cfg(feature = "std")]
    fn bytes(&self) -> usize {
        2 + (self.ntrans * 2) + (self.ntrans * size_of::<S>())
    }
}

#[cfg(feature = "std")]
impl<'a, S: StateID> fmt::Debug for State<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut transitions = vec![];
        for i in 0..(self.ntrans - 1) {
            let next = self.next_at(i);
            if next == dead_id() {
                continue;
            }

            let (start, end) = self.range(i);
            if start == end {
                transitions.push(format!(
                    "{} => {}",
                    escape(start),
                    next.as_usize()
                ));
            } else {
                transitions.push(format!(
                    "{}-{} => {}",
                    escape(start),
                    escape(end),
                    next.as_usize(),
                ));
            }
        }
        let eof = self.next_at(self.ntrans - 1);
        if eof != dead_id() {
            transitions.push(format!("EOF => {}", eof.as_usize()));
        }
        write!(f, "{}", transitions.join(", "))
    }
}

/// A representation of a mutable sparse DFA state that can be cheaply
/// materialized from a state identifier.
#[cfg(feature = "std")]
struct StateMut<'a, S> {
    /// The state identifier representation used by the DFA from which this
    /// state was extracted. Since our transition table is compacted in a
    /// &[u8], we don't actually use the state ID type parameter explicitly
    /// anywhere, so we fake it. This prevents callers from using an incorrect
    /// state ID representation to read from this state.
    _state_id_repr: PhantomData<S>,
    /// The number of transitions in this state.
    ntrans: usize,
    /// Pairs of input ranges, where there is one pair for each transition.
    /// Each pair specifies an inclusive start and end byte range for the
    /// corresponding transition.
    input_ranges: &'a mut [u8],
    /// Transitions to the next state. This slice contains native endian
    /// encoded state identifiers, with `S` as the representation. Thus, there
    /// are `ntrans * size_of::<S>()` bytes in this slice.
    next: &'a mut [u8],
}

#[cfg(feature = "std")]
impl<'a, S: StateID> StateMut<'a, S> {
    /// Sets the ith transition to the given state.
    fn set_next_at(&mut self, i: usize, next: S) {
        next.write_bytes(&mut self.next[i * size_of::<S>()..]);
    }
}

#[cfg(feature = "std")]
impl<'a, S: StateID> fmt::Debug for StateMut<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = State {
            _state_id_repr: self._state_id_repr,
            ntrans: self.ntrans,
            input_ranges: self.input_ranges,
            next: self.next,
        };
        fmt::Debug::fmt(&state, f)
    }
}

/// Convert the given `usize` to the chosen state identifier
/// representation. If the given value cannot fit in the chosen
/// representation, then an error is returned.
#[cfg(feature = "std")]
fn usize_to_state_id<S: StateID>(value: usize) -> Result<S, Error> {
    if value > S::max_id() {
        Err(Error::state_id_overflow(S::max_id()))
    } else {
        Ok(S::from_usize(value))
    }
}

/// Return the given byte as its escaped string form.
#[cfg(feature = "std")]
fn escape(b: u8) -> String {
    use std::ascii;

    String::from_utf8(ascii::escape_default(b).collect::<Vec<_>>()).unwrap()
}

/// A binary search routine specialized specifically to a sparse DFA state's
/// transitions. Specifically, the transitions are defined as a set of pairs
/// of input bytes that delineate an inclusive range of bytes. If the input
/// byte is in the range, then the corresponding transition is a match.
///
/// This binary search accepts a slice of these pairs and returns the position
/// of the matching pair (the ith transition), or None if no matching pair
/// could be found.
///
/// Note that this routine is not currently used since it was observed to
/// either decrease performance when searching ASCII, or did not provide enough
/// of a boost on non-ASCII haystacks to be worth it. However, we leave it here
/// for posterity in case we can find a way to use it.
///
/// In theory, we could use the standard library's search routine if we could
/// cast a `&[u8]` to a `&[(u8, u8)]`, but I don't believe this is currently
/// guaranteed to be safe and is thus UB (since I don't think the in-memory
/// representation of `(u8, u8)` has been nailed down).
#[inline(always)]
#[allow(dead_code)]
fn binary_search_ranges(ranges: &[u8], needle: u8) -> Option<usize> {
    debug_assert!(ranges.len() % 2 == 0, "ranges must have even length");
    debug_assert!(ranges.len() <= 512, "ranges should be short");

    let (mut left, mut right) = (0, ranges.len() / 2);
    while left < right {
        let mid = (left + right) / 2;
        let (b1, b2) = (ranges[mid * 2], ranges[mid * 2 + 1]);
        if needle < b1 {
            right = mid;
        } else if needle > b2 {
            left = mid + 1;
        } else {
            return Some(mid);
        }
    }
    None
}
