"""
external_declaration.py

Record what correction was applied to a DWI *before* csttool received it.

csttool skips its own preprocessing unless asked, so most runs operate on data
that some other tool has already corrected. That was previously an implicit
assumption baked into a metadata string ("Skipped (External Preprocessing
Used)") which asserted, as fact, something nobody had stated and csttool cannot
check. ``--input-corrected`` replaces the assumption with a declaration.

Every value is a **user declaration**. csttool does not inspect the data to
confirm it, does not infer it from BIDS metadata, and does not change any
processing decision because of it. The only thing it acts on is one advisory
warning (see :func:`motion_correction_conflict_warning`).
"""

# Ordered as they appear in --help. `unknown` is the default so that
# "nobody said" stays distinguishable from "the user declared that nothing was
# done" — collapsing those two is what made the old metadata misleading.
EXTERNAL_CORRECTION_CHOICES = (
    "unknown",
    "none",
    "topup-eddy",
    "eddy-only",
    "other",
)

#: Declarations that assert an external motion-correction-class step.
_DECLARES_MOTION_CORRECTION = ("topup-eddy", "eddy-only")

DEFAULT_EXTERNAL_CORRECTION = "unknown"


def declaration_record(declared: str | None) -> dict:
    """Build the provenance record for an external-correction declaration.

    ``verified_by_csttool`` is hard-coded ``False`` and has no code path that
    can set it ``True``: nothing in csttool can verify this claim, and a field
    that is always False is more honest than one that looks settable.
    """
    value = declared or DEFAULT_EXTERNAL_CORRECTION
    if value not in EXTERNAL_CORRECTION_CHOICES:
        raise ValueError(
            f"Unknown external-correction declaration {value!r}; "
            f"expected one of {list(EXTERNAL_CORRECTION_CHOICES)}"
        )
    return {
        "declared": value,
        "verified_by_csttool": False,
        "source": "user-declaration",
    }


def motion_correction_conflict_warning(
    declared: str | None, apply_motion_correction: bool
) -> str | None:
    """Warn when the input is declared motion-corrected and MC is also requested.

    Returns the warning text, or ``None`` when there is nothing to say.

    Deliberately a warning rather than a hard error. The declaration is
    unverified user input, so it must not be treated as stronger evidence than
    an explicit flag: a mistyped or over-broad declaration would otherwise
    block a legitimate run. Running two motion corrections is usually
    undesirable — the second one re-estimates and re-resamples data that is
    already aligned, adding interpolation blur for no gain — but it is not
    intrinsically invalid, and a user may well want to measure the residual
    motion an external tool left behind.
    """
    if not apply_motion_correction:
        return None
    if (declared or DEFAULT_EXTERNAL_CORRECTION) not in _DECLARES_MOTION_CORRECTION:
        return None
    return (
        f"Likely double motion correction: the input is declared as already "
        f"corrected ('{declared}') and --perform-motion-correction was also "
        f"requested. csttool will run its own affine motion correction anyway "
        f"(the declaration is unverified), which resamples already-aligned data "
        f"a second time. Drop one of the two unless this is intentional."
    )


def passthrough_advisory(declared: str | None) -> str | None:
    """Advisory for a pass-through run whose input history was never declared.

    Returned only when csttool did no preprocessing of its own *and* the user
    made no declaration: in that case the outputs rest on an assumption the
    report cannot substantiate. Print-only; csttool processes what it is given.
    """
    value = declared or DEFAULT_EXTERNAL_CORRECTION
    if value != "unknown":
        return None
    return (
        "csttool ran no preprocessing of its own and the input's correction "
        "history was not declared, so these results assume the data was already "
        "corrected elsewhere.\n"
        "  Declare it with --input-corrected {none,topup-eddy,eddy-only,other}, "
        "or pass --preprocess to let csttool denoise and mask."
    )
