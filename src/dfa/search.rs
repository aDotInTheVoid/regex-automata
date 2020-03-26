use crate::dfa::automaton::Automaton;
use crate::NoMatch;

pub fn find_earliest_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<usize>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = init_fwd(dfa, bytes, start, end)?;
    if last_match.is_some() {
        return Ok(last_match);
    }
    unsafe {
        let mut p = bytes.as_ptr().add(start);
        while p < bytes[end..].as_ptr() {
            let byte = *p;
            state = dfa.next_state_unchecked(state, byte);
            p = p.add(1);
            if dfa.is_special_state(state) {
                return if dfa.is_dead_state(state) {
                    Ok(None)
                } else if dfa.is_quit_state(state) {
                    Err(NoMatch::Quit { byte, offset: offset(bytes, p) - 1 })
                } else {
                    Ok(Some(offset(bytes, p) - dfa.match_offset()))
                };
            }
        }
    }
    eof_fwd(dfa, bytes, end, &mut state)
}

pub fn find_earliest_rev<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<usize>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = init_rev(dfa, bytes, start, end)?;
    if last_match.is_some() {
        return Ok(last_match);
    }
    unsafe {
        let mut p = bytes.as_ptr().add(end);
        while p > bytes[start..].as_ptr() {
            p = p.sub(1);
            let byte = *p;
            state = dfa.next_state_unchecked(state, byte);
            if dfa.is_special_state(state) {
                return if dfa.is_dead_state(state) {
                    Ok(None)
                } else if dfa.is_quit_state(state) {
                    Err(NoMatch::Quit { byte, offset: offset(bytes, p) })
                } else {
                    Ok(Some(offset(bytes, p) + dfa.match_offset()))
                };
            }
        }
    }
    eof_rev(dfa, state, bytes, start)
}

pub fn find_leftmost_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<usize>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = init_fwd(dfa, bytes, start, end)?;
    unsafe {
        let mut p = bytes.as_ptr().add(start);
        while p < bytes[end..].as_ptr() {
            let byte = *p;
            state = dfa.next_state_unchecked(state, byte);
            p = p.add(1);
            if dfa.is_special_state(state) {
                if dfa.is_dead_state(state) {
                    return Ok(last_match);
                } else if dfa.is_quit_state(state) {
                    return Err(NoMatch::Quit {
                        byte,
                        offset: offset(bytes, p) - 1,
                    });
                }
                last_match = Some(offset(bytes, p) - dfa.match_offset());
            }
        }
    }
    Ok(eof_fwd(dfa, bytes, end, &mut state)?.or(last_match))
}

pub fn find_leftmost_rev<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<Option<usize>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = init_rev(dfa, bytes, start, end)?;
    unsafe {
        let mut p = bytes.as_ptr().add(end);
        while p > bytes[start..].as_ptr() {
            p = p.sub(1);
            let byte = *p;
            state = dfa.next_state_unchecked(state, byte);
            if dfa.is_special_state(state) {
                if dfa.is_dead_state(state) {
                    return Ok(last_match);
                } else if dfa.is_quit_state(state) {
                    return Err(NoMatch::Quit {
                        byte,
                        offset: offset(bytes, p),
                    });
                }
                last_match = Some(offset(bytes, p) + dfa.match_offset());
            }
        }
    }
    Ok(eof_rev(dfa, state, bytes, start)?.or(last_match))
}

pub fn find_overlapping_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    mut start: usize,
    end: usize,
    state_id: &mut Option<A::ID>,
) -> Result<Option<usize>, NoMatch> {
    assert!(start <= end);
    assert!(start <= bytes.len());
    assert!(end <= bytes.len());

    let (mut state, mut last_match) = match *state_id {
        None => init_fwd(dfa, bytes, start, end)?,
        Some(id) => {
            // This is a subtle but critical detail. If the caller provides a
            // non-None state ID, then it must be the case that the state ID
            // corresponds to one set by this function. The state ID therefore
            // corresponds to a match state, a dead state or some other state.
            // However, "some other" state _only_ occurs when the input has
            // been exhausted because the only way to stop before then is to
            // see a match or a dead state.
            //
            // If the input is exhausted or if it's a dead state, then
            // incrementing the starting position has no relevance on
            // correctness, since the loop below with either not execute
            // at all or will immediate stop due to being in a dead state.
            // (Once in a dead state it is impossible to leave it.)
            //
            // Therefore, the only case we need to consider is when state_id
            // is a match state. In this case, since our machines support the
            // ability to delay a match by a certain number of bytes (to
            // support look-around), it follows that we actually consumed
            // that many additional bytes on our previous search. When the
            // caller resumes their search to find subsequent matches, they
            // will use the ending location from the previous match as the
            // next starting point, which is `match_offset` bytes PRIOR to
            // where we scanned to on the previous search. Therefore, we need
            // to compensate by bumping `start` up by `match_offset` bytes.
            start += dfa.match_offset();
            // Since match_offset could be any arbitrary value and we use
            // `start` in pointer arithmetic below, we check that we are still
            // in bounds. Otherwise, we could materialize a pointer that is
            // more than one past the end point of `bytes`, which is UB.
            if start > end {
                return Ok(None);
            }
            (id, None)
        }
    };
    if last_match.is_some() {
        *state_id = Some(state);
        return Ok(last_match);
    }
    // TODO: Do not use unsafe here.
    unsafe {
        let mut p = bytes.as_ptr().add(start);
        while p < bytes[end..].as_ptr() {
            let byte = *p;
            state = dfa.next_state_unchecked(state, byte);
            p = p.add(1);
            if dfa.is_special_state(state) {
                *state_id = Some(state);
                return if dfa.is_dead_state(state) {
                    Ok(None)
                } else if dfa.is_quit_state(state) {
                    Err(NoMatch::Quit { byte, offset: offset(bytes, p) - 1 })
                } else {
                    Ok(Some(offset(bytes, p) - dfa.match_offset()))
                };
            }
        }
    }
    let result = eof_fwd(dfa, bytes, end, &mut state);
    *state_id = Some(state);
    result
}

fn init_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<(A::ID, Option<usize>), NoMatch> {
    let state = dfa.start_state_forward(bytes, start, end);
    if dfa.is_match_state(state) {
        Ok((state, Some(start - dfa.match_offset())))
    } else {
        Ok((state, None))
    }
}

fn init_rev<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<(A::ID, Option<usize>), NoMatch> {
    let state = dfa.start_state_reverse(bytes, start, end);
    if dfa.is_match_state(state) {
        Ok((state, Some(end + dfa.match_offset())))
    } else {
        Ok((state, None))
    }
}

fn eof_fwd<A: Automaton + ?Sized>(
    dfa: &A,
    bytes: &[u8],
    end: usize,
    state: &mut A::ID,
) -> Result<Option<usize>, NoMatch> {
    match bytes.get(end) {
        Some(&b) => {
            *state = dfa.next_state(*state, b);
            if dfa.is_match_state(*state) {
                Ok(Some(end))
            } else {
                Ok(None)
            }
        }
        None => {
            *state = dfa.next_eof_state(*state);
            if dfa.is_match_state(*state) {
                Ok(Some(bytes.len()))
            } else {
                Ok(None)
            }
        }
    }
}

fn eof_rev<A: Automaton + ?Sized>(
    dfa: &A,
    state: A::ID,
    bytes: &[u8],
    start: usize,
) -> Result<Option<usize>, NoMatch> {
    if start > 0 {
        if dfa.is_match_state(dfa.next_state(state, bytes[start - 1])) {
            Ok(Some(start))
        } else {
            Ok(None)
        }
    } else {
        if dfa.is_match_state(dfa.next_eof_state(state)) {
            Ok(Some(0))
        } else {
            Ok(None)
        }
    }
}

/// Returns the distance between the given pointer and the start of `bytes`.
/// This assumes that the given pointer points to somewhere in the `bytes`
/// slice given.
fn offset(bytes: &[u8], p: *const u8) -> usize {
    ((p as isize) - (bytes.as_ptr() as isize)) as usize
}
