use std::error::Error;

use regex_automata::{
    dfa::{dense, Automaton},
    nfa::thompson,
    NoMatch,
};

#[test]
fn quit_fwd() -> Result<(), Box<dyn Error>> {
    let dfa = dense::Builder::new()
        .configure(dense::Config::new().quit(b'x', true))
        .build("[[:word:]]+$")?;

    assert_eq!(
        dfa.find_earliest_fwd(b"abcxyz"),
        Err(NoMatch::Quit { byte: b'x', offset: 3 })
    );
    assert_eq!(
        dfa.find_leftmost_fwd(b"abcxyz"),
        Err(NoMatch::Quit { byte: b'x', offset: 3 })
    );
    assert_eq!(
        dfa.find_overlapping_fwd(b"abcxyz", &mut None),
        Err(NoMatch::Quit { byte: b'x', offset: 3 })
    );

    Ok(())
}

#[test]
fn quit_rev() -> Result<(), Box<dyn Error>> {
    let dfa = dense::Builder::new()
        .configure(dense::Config::new().quit(b'x', true))
        .thompson(thompson::Config::new().reverse(true))
        .build("^[[:word:]]+")?;

    assert_eq!(
        dfa.find_earliest_rev(b"abcxyz"),
        Err(NoMatch::Quit { byte: b'x', offset: 3 })
    );
    assert_eq!(
        dfa.find_leftmost_rev(b"abcxyz"),
        Err(NoMatch::Quit { byte: b'x', offset: 3 })
    );

    Ok(())
}

#[test]
#[should_panic]
fn quit_panics() {
    dense::Config::new().unicode_word_boundary(true).quit(b'\xFF', false);
}

// This tests an intesting case where even if the Unicode word boundary option
// is disabled, setting all non-ASCII bytes to be quit bytes will cause Unicode
// word boundaries to be enabled.
#[test]
fn unicode_word_implicitly_works() -> Result<(), Box<dyn Error>> {
    let mut config = dense::Config::new();
    for b in 0x80..=0xFF {
        config = config.quit(b, true);
    }
    let dfa = dense::Builder::new().configure(config).build(r"\b")?;
    assert_eq!(dfa.find_leftmost_fwd(b" a"), Ok(Some(1)),);
    Ok(())
}
