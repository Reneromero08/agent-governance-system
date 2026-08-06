# Source Reproduction Report V3

Status: source package reproduction only. Scientific verification is separate.

Scientific source SHA: `7c79414f2beb34c29bf0d63783a6effea26c65ed`
Source worktree: `/tmp/ags-audio-source-7c79414-v3`
Source status before: `clean`
Source status after: `clean`

## Classifications

- Candidate I: `SOURCE_REPRODUCED`
- Candidate J: `SOURCE_REPRODUCED`
- Candidate K: `SOURCE_REPRODUCED`
- Candidate L: `SOURCE_REPRODUCED` — corrective primary-result marker stale control passed closed
- Candidate M: `SOURCE_REPRODUCED`
- Candidate N: `SOURCE_NOT_REPRODUCED` — long evidence-root run failed before bind; short-path control reproduced twice
- Candidate O: `SOURCE_REPRODUCED`

## Runs

- I run1: rc=0, files=77, stdout=433464b3869a0f3b1d513657567be490c243199d1a70eef3c81e31968fe80726, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- I run2: rc=0, files=77, stdout=433464b3869a0f3b1d513657567be490c243199d1a70eef3c81e31968fe80726, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- J run1: rc=0, files=130, stdout=9483d9ef3ccb26959f38590fc92ef9e38a4e7bfd9977b3a62008959af7bedafd, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- J run2: rc=0, files=130, stdout=9483d9ef3ccb26959f38590fc92ef9e38a4e7bfd9977b3a62008959af7bedafd, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- K run1: rc=0, files=396, stdout=f8318aeda8639fec8851afb20e3b61bc85739adfe4e9eb34c28404c4538cf758, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- K run2: rc=0, files=396, stdout=f8318aeda8639fec8851afb20e3b61bc85739adfe4e9eb34c28404c4538cf758, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- L run1: rc=0, files=580, stdout=3371056ff0f918b44f1d740b735d98c02da61eea708c96e12240bc9ff614aa9c, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- L run2: rc=0, files=580, stdout=3371056ff0f918b44f1d740b735d98c02da61eea708c96e12240bc9ff614aa9c, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- M run1: rc=0, files=129, stdout=784c1dc0f4e4d02f932f19137691d3a646eb65525e24a07a423fc0a8b104e64e, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- M run2: rc=0, files=129, stdout=784c1dc0f4e4d02f932f19137691d3a646eb65525e24a07a423fc0a8b104e64e, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- N run1: rc=1, files=7, stdout=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- N run2: rc=1, files=7, stdout=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- O run1: rc=0, files=74, stdout=142a69849d02a6a4dbd80d5242d8a471b011b825dd3dfbd49f2ff71e3e1eb549, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- O run2: rc=0, files=74, stdout=142a69849d02a6a4dbd80d5242d8a471b011b825dd3dfbd49f2ff71e3e1eb549, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

## Controls

- I missing-argument control: rc=2
- J missing-argument control: rc=2
- K missing-argument control: rc=2
- L missing-argument control: rc=2
- M missing-argument control: rc=2
- N missing-argument control: rc=2
- O missing-argument control: rc=2
- I generic stale `result.json` control: overwritten_or_rejected=True, rc=0
- J generic stale `result.json` control: overwritten_or_rejected=True, rc=0
- K generic stale `result.json` control: overwritten_or_rejected=True, rc=0
- L generic stale `result.json` control: overwritten_or_rejected=False, rc=0
- M generic stale `result.json` control: overwritten_or_rejected=True, rc=0
- N generic stale `result.json` control: overwritten_or_rejected=False, rc=1
- O generic stale `result.json` control: overwritten_or_rejected=True, rc=0

## Corrected candidate-specific controls

- Candidate L primary stale `result.full.json`: rc=0, overwritten=True
- Candidate L corrective marker `result.full.json`: rc=0, pre/post differ=True, marker absent=True, schema passed=True
- Candidate N short-path reproduction: rc1=0, rc2=0, copied_path=`raw_outputs/source_reproduction_v3_short_path_controls/n_short_reproduction`

## Interpretation notes

- Candidate L does not use `result.json` as its primary generated result; it writes `result.full.json`. The corrected primary-result stale control overwrote the stale file and returned success, so L is not treated as fail-open on that basis.
- The corrective Candidate L marker control records a pre-run impossible-marker hash and a post-run result hash. L source reproduction is counted as closed for this specific stale-output control only if the hashes differ, the marker disappears, and the generated result parses with the expected primary schema.
- Candidate N failed in the deep V3 evidence root before socket bind, but reproduced twice in a short `/tmp` output path. This is recorded as path-depth sensitivity of the qualifier/protocol package rather than ordinary semantic reproduction under the long evidence path.
