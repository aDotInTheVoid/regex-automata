use regex_automata::dfa::{dense, sparse, Automaton, Regex, RegexBuilder};
use regex_automata::nfa::thompson;
use regex_automata::{MatchKind, SyntaxConfig};
use regex_syntax as syntax;

use regex_test::bstr::{BString, ByteSlice};
use regex_test::{
    CompiledRegex, Match, MatchKind as TestMatchKind, RegexTest, RegexTests,
    SearchKind as TestSearchKind, TestResult, TestRunner,
};

use crate::{suite, Result};

#[test]
fn unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    TestRunner::new()?
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn unminimized_no_byte_class() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.dense(dense::Config::new().byte_classes(false));

    TestRunner::new()?
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn unminimized_no_nfa_shrink_default() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.thompson(thompson::Config::new().shrink(false));

    TestRunner::new()?
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn minimized_default() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.dense(dense::Config::new().minimize(true));
    TestRunner::new()?
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn minimized_no_byte_class() -> Result<()> {
    let mut builder = RegexBuilder::new();
    builder.dense(dense::Config::new().minimize(true).byte_classes(false));

    TestRunner::new()?
        .blacklist("expensive")
        .test_iter(suite()?.iter(), dense_compiler(builder))
        .assert();
    Ok(())
}

#[test]
fn sparse_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    TestRunner::new()?
        .test_iter(suite()?.iter(), sparse_compiler(builder))
        .assert();
    Ok(())
}

// A basic sanity test that checks we can convert a regex to a smaller
// representation and that the resulting regex still passes our tests.
//
// If tests grow minimal regexes that cannot be represented in 16 bits, then
// we'll either want to skip those or increase the size to test to u32.
#[test]
fn u16_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    TestRunner::new()?
        // Some of these are too big to fit into u16 state IDs.
        .blacklist("expensive")
        // These too. Because \w is gigantic.
        .blacklist("bytes/perl-word-unicode")
        .blacklist("unicode/perl")
        .blacklist("no-unicode/word-unicode")
        .test_iter(suite()?.iter(), u16_compiler(builder))
        .assert();
    Ok(())
}

// Test that sparse DFAs work after converting them to a different state ID
// representation.
#[test]
fn sparse_u16_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    TestRunner::new()?
        // Some of these are too big to fit into u16 state IDs.
        .blacklist("expensive")
        // These too. Because \w is gigantic.
        .blacklist("bytes/perl-word-unicode")
        .blacklist("unicode/perl")
        .blacklist("no-unicode/word-unicode")
        .test_iter(suite()?.iter(), sparse_u16_compiler(builder))
        .assert();
    Ok(())
}

// Another basic sanity test that checks we can serialize and then deserialize
// a regex, and that the resulting regex can be used for searching correctly.
#[test]
fn serialization_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    let my_compiler = |builder| {
        compiler(builder, |re| {
            let fwd_bytes = re.forward().to_bytes_native_endian()?;
            let rev_bytes = re.reverse().to_bytes_native_endian()?;
            Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
                let fwd: dense::DFA<&[usize], usize> =
                    unsafe { dense::DFA::from_bytes_unchecked(&fwd_bytes) };
                let rev: dense::DFA<&[usize], usize> =
                    unsafe { dense::DFA::from_bytes_unchecked(&rev_bytes) };
                let re = Regex::from_dfas(fwd, rev);

                run_test(&re, test)
            }))
        })
    };
    TestRunner::new()?
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}

// A basic sanity test that checks we can serialize and then deserialize a
// regex using sparse DFAs, and that the resulting regex can be used for
// searching correctly.
#[test]
fn sparse_serialization_unminimized_default() -> Result<()> {
    let builder = RegexBuilder::new();
    let my_compiler = |builder| {
        compiler(builder, |re| {
            let fwd_bytes =
                re.forward().to_sparse()?.to_bytes_native_endian()?;
            let rev_bytes =
                re.reverse().to_sparse()?.to_bytes_native_endian()?;
            Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
                let fwd: sparse::DFA<&[u8], usize> =
                    unsafe { sparse::DFA::from_bytes_unchecked(&fwd_bytes) };
                let rev: sparse::DFA<&[u8], usize> =
                    unsafe { sparse::DFA::from_bytes_unchecked(&rev_bytes) };
                let re = Regex::from_dfas(fwd, rev);

                run_test(&re, test)
            }))
        })
    };
    TestRunner::new()?
        .test_iter(suite()?.iter(), my_compiler(builder))
        .assert();
    Ok(())
}

fn dense_compiler(
    builder: RegexBuilder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |re| {
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, test)
        }))
    })
}

fn sparse_compiler(
    builder: RegexBuilder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |re| {
        let fwd = re.forward().to_sparse()?;
        let rev = re.reverse().to_sparse()?;
        let re = Regex::from_dfas(fwd, rev);
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, test)
        }))
    })
}

fn u16_compiler(
    builder: RegexBuilder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |re| {
        let fwd = re.forward().to_u16()?;
        let rev = re.reverse().to_u16()?;
        let re = Regex::from_dfas(fwd, rev);
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, test)
        }))
    })
}

fn sparse_u16_compiler(
    builder: RegexBuilder,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    compiler(builder, |re| {
        let fwd = re.forward().to_sparse()?.to_u16()?;
        let rev = re.reverse().to_sparse()?.to_u16()?;
        let re = Regex::from_dfas(fwd, rev);
        Ok(CompiledRegex::compiled(move |test| -> Vec<TestResult> {
            run_test(&re, test)
        }))
    })
}

fn compiler(
    mut builder: RegexBuilder,
    mut create_matcher: impl FnMut(Regex) -> Result<CompiledRegex>,
) -> impl FnMut(&RegexTest, &[BString]) -> Result<CompiledRegex> {
    move |test, regexes| {
        if regexes.len() != 1 {
            return Ok(CompiledRegex::skip());
        }
        let regex = regexes[0].to_str()?;

        // Check if our regex contains things that aren't supported by DFAs.
        // That is, Unicode word boundaries when searching non-ASCII text.
        let mut thompson = thompson::Builder::new();
        thompson.configure(config_thompson(test));
        if let Ok(nfa) = thompson.build(regex) {
            let non_ascii = test.input().iter().any(|&b| !b.is_ascii());
            if nfa.has_word_boundary_unicode() && non_ascii {
                return Ok(CompiledRegex::skip());
            }
        }
        if !configure_regex_builder(test, &mut builder) {
            return Ok(CompiledRegex::skip());
        }
        create_matcher(builder.build(regex)?)
    }
}

fn run_test<A: Automaton>(re: &Regex<A>, test: &RegexTest) -> Vec<TestResult> {
    let is_match = if re.is_match(test.input()) {
        TestResult::matched()
    } else {
        TestResult::no_match()
    };
    let is_match = is_match.name("is_match");

    let find_matches = match test.search_kind() {
        TestSearchKind::Earliest => {
            let it = re
                .find_earliest_iter(test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match { start: m.start(), end: m.end() });
            TestResult::matches(it).name("find_earliest_iter")
        }
        TestSearchKind::Leftmost => {
            let it = re
                .find_leftmost_iter(test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match { start: m.start(), end: m.end() });
            TestResult::matches(it).name("find_leftmost_iter")
        }
        TestSearchKind::Overlapping => {
            let it = re
                .find_overlapping_iter(test.input())
                .take(test.match_limit().unwrap_or(std::usize::MAX))
                .map(|m| Match { start: m.start(), end: m.end() });
            TestResult::matches(it).name("find_overlapping_iter")
        }
    };

    vec![is_match, find_matches]
}

fn configure_regex_builder(
    test: &RegexTest,
    builder: &mut RegexBuilder,
) -> bool {
    let match_kind = match test.match_kind() {
        TestMatchKind::All => MatchKind::All,
        TestMatchKind::LeftmostFirst => MatchKind::LeftmostFirst,
        TestMatchKind::LeftmostLongest => return false,
    };

    let syntax_config = SyntaxConfig::new()
        .case_insensitive(test.case_insensitive())
        .unicode(test.unicode())
        .allow_invalid_utf8(!test.utf8());
    let dense_config = dense::Config::new()
        .anchored(test.anchored())
        .match_kind(match_kind)
        .unicode_word_boundary(true);

    builder.syntax(syntax_config).dense(dense_config);
    true
}

fn config_thompson(_: &RegexTest) -> thompson::Config {
    // Currently there are no config options that influence Thompson
    // construction.
    thompson::Config::new()
}
