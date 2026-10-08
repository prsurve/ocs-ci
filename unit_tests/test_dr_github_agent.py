"""
Unit tests for DR GitHub Agent (DRGitHubAgent)

Covers:
  - RDR keyword filtering (_is_rdr_related)
  - PR status derivation (_pr_status)
  - Age calculation (_age_days)
  - Review summary aggregation (_fetch_reviews)
  - CI check-run status rollup (_fetch_check_status)
  - Full enrichment (_enrich_pr)
  - DRGitHubAgent.list_rdr_prs — end-to-end with mocked HTTP
  - DRGitHubAgent.print_table / print_summary / print_detail (smoke)
  - Token resolution (_resolve_token)
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from ocs_ci.ocs.dr.dr_github_agent import (
    DRGitHubAgent,
    RDR_LABELS,
    _age_days,
    _enrich_pr,
    _fetch_check_status,
    _fetch_files_changed,
    _fetch_reviews,
    _is_rdr_related,
    _pr_status,
    _resolve_token,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_pr(
    number: int = 1001,
    title: str = "feat: rdr failover support",
    state: str = "open",
    draft: bool = False,
    merged_at: str = None,
    labels: list = None,
    assignees: list = None,
    requested_reviewers: list = None,
    milestone: str = None,
    created_at: str = None,
    updated_at: str = None,
    head_ref: str = "rdr/failover-fix",
    base_ref: str = "master",
    author: str = "dev-user",
    comments: int = 2,
    review_comments: int = 1,
) -> dict:
    """Build a minimal GitHub PR payload dict."""
    now = datetime.now(timezone.utc)
    created = created_at or (now - timedelta(days=5)).isoformat()
    updated = updated_at or now.isoformat()

    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/red-hat-storage/ocs-ci/pull/{number}",
        "state": state,
        "draft": draft,
        "merged_at": merged_at,
        "created_at": created,
        "updated_at": updated,
        "comments": comments,
        "review_comments": review_comments,
        "user": {"login": author},
        "labels": [{"name": lbl} for lbl in (labels or [])],
        "assignees": [{"login": a} for a in (assignees or [])],
        "requested_reviewers": [{"login": r} for r in (requested_reviewers or [])],
        "milestone": {"title": milestone} if milestone else None,
        "head": {"ref": head_ref, "sha": "abc123"},
        "base": {"ref": base_ref},
    }


def _make_review(login: str, state: str) -> dict:
    return {"user": {"login": login}, "state": state, "body": ""}


def _make_check_run(status: str = "completed", conclusion: str = "success") -> dict:
    return {"status": status, "conclusion": conclusion, "name": "ci-check"}


# ── _is_rdr_related ───────────────────────────────────────────────────────────


class TestIsRdrRelated:
    def test_matches_title_rdr(self):
        assert _is_rdr_related(_make_pr(title="Add RDR failover logic"))

    def test_matches_title_regional_dr(self):
        assert _is_rdr_related(_make_pr(title="Fix regional-dr topology view"))

    def test_matches_title_disaster_recovery(self):
        assert _is_rdr_related(_make_pr(title="Disaster Recovery e2e test"))

    def test_matches_label(self):
        pr = _make_pr(
            title="Update docs", labels=["disaster-recovery"], head_ref="chore/docs"
        )
        assert _is_rdr_related(pr)

    def test_matches_milestone(self):
        pr = _make_pr(title="Update docs", milestone="RDR 4.17", head_ref="chore/docs")
        assert _is_rdr_related(pr)

    def test_matches_head_branch(self):
        pr = _make_pr(title="Fix something", head_ref="rdr/fix-timeout")
        assert _is_rdr_related(pr)

    def test_matches_drpolicy_keyword(self):
        assert _is_rdr_related(_make_pr(title="Add DRPolicy validation"))

    def test_no_match_unrelated_pr(self):
        pr = _make_pr(
            title="Fix MCG bucket replication", head_ref="mcg/fix", labels=["mcg"]
        )
        assert not _is_rdr_related(pr)

    def test_case_insensitive(self):
        assert _is_rdr_related(_make_pr(title="REGIONAL DR FAILOVER"))

    def test_matches_volsync(self):
        assert _is_rdr_related(_make_pr(title="volsync integration tests"))

    def test_matches_odr(self):
        assert _is_rdr_related(_make_pr(title="ODR 5m policy e2e"))

    # ── New cases covering PR #16069 and Squad/Turquoise ─────────────────────

    def test_matches_squad_turquoise_label(self):
        """PR #16069: Squad/Turquoise label alone must be enough to match."""
        pr = _make_pr(
            title="RHSTOR-8222/OCSQE-5015: Add OpenShift Lightspeed (OLS) DR Recipe generation support",
            labels=["Verified", "size/XXL", "Squad/Turquoise"],
            head_ref="recipe_generation_ols",
        )
        assert _is_rdr_related(pr)

    def test_squad_turquoise_label_case_insensitive(self):
        """Label matching must be case-insensitive (GitHub can return any casing)."""
        pr = _make_pr(
            title="chore: unrelated", labels=["SQUAD/TURQUOISE"], head_ref="chore/x"
        )
        assert _is_rdr_related(pr)

    def test_squad_turquoise_label_no_keywords_needed(self):
        """Even a completely neutral title/branch matches via Squad/Turquoise."""
        pr = _make_pr(
            title="chore: update docs",
            labels=["Squad/Turquoise"],
            head_ref="chore/docs",
        )
        assert _is_rdr_related(pr)

    def test_dr_word_boundary_in_title(self):
        """Standalone 'DR' surrounded by spaces must match (' dr ' keyword)."""
        pr = _make_pr(
            title="Add OpenShift Lightspeed (OLS) DR Recipe generation support",
            labels=[],
            head_ref="recipe_generation_ols",
        )
        assert _is_rdr_related(pr)

    def test_dr_word_at_end_of_title(self):
        """'DR' at the end of a title (no trailing space in original) must match."""
        pr = _make_pr(title="Improve support for ODR", head_ref="fix/x", labels=[])
        assert _is_rdr_related(pr)

    def test_no_false_positive_address(self):
        """'address' contains 'dr' but must NOT match — only space-bounded ' dr ' does."""
        pr = _make_pr(
            title="Refactor to address flaky test",
            head_ref="fix/flaky",
            labels=["flaky"],
        )
        assert not _is_rdr_related(pr)

    def test_no_false_positive_dryrun(self):
        """'dryrun' contains 'dr' but must NOT trigger the space-padded ' dr ' match."""
        pr = _make_pr(
            title="Add dryrun support for deploy", head_ref="feat/deploy", labels=[]
        )
        assert not _is_rdr_related(pr)

    def test_rdr_labels_set_contains_turquoise(self):
        """Verify the exported RDR_LABELS constant includes squad/turquoise."""
        assert "squad/turquoise" in RDR_LABELS

    def test_matches_recipe_keyword(self):
        """'recipe' keyword catches DR recipe / runbook PRs."""
        pr = _make_pr(
            title="Add DR recipe for busybox workload",
            labels=[],
            head_ref="feat/recipe",
        )
        assert _is_rdr_related(pr)


# ── _pr_status ────────────────────────────────────────────────────────────────


class TestPrStatus:
    def test_open(self):
        assert _pr_status(_make_pr(state="open", draft=False)) == "open"

    def test_draft(self):
        assert _pr_status(_make_pr(state="open", draft=True)) == "draft"

    def test_merged(self):
        pr = _make_pr(state="closed", merged_at="2024-01-01T00:00:00Z")
        assert _pr_status(pr) == "merged"

    def test_closed(self):
        pr = _make_pr(state="closed", merged_at=None)
        assert _pr_status(pr) == "closed"


# ── _age_days ─────────────────────────────────────────────────────────────────


class TestAgeDays:
    def test_zero_days(self):
        now = datetime.now(timezone.utc).isoformat()
        assert _age_days(now) == 0

    def test_five_days(self):
        five_ago = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        assert _age_days(five_ago) == 5

    def test_thirty_days(self):
        thirty_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert _age_days(thirty_ago) == 30


# ── _fetch_reviews ────────────────────────────────────────────────────────────


class TestFetchReviews:
    def _patch_paginate(self, reviews):
        return patch(
            "ocs_ci.ocs.dr.dr_github_agent._paginate",
            return_value=reviews,
        )

    def test_approved(self):
        with self._patch_paginate([_make_review("alice", "APPROVED")]):
            result = _fetch_reviews(42, token=None)
        assert result["decision"] == "APPROVED"
        assert "alice" in result["approved_by"]
        assert result["changes_by"] == []

    def test_changes_requested(self):
        with self._patch_paginate([_make_review("bob", "CHANGES_REQUESTED")]):
            result = _fetch_reviews(42, token=None)
        assert result["decision"] == "CHANGES_REQUESTED"
        assert "bob" in result["changes_by"]

    def test_mixed_latest_wins(self):
        """When a reviewer first requests changes then approves, final should be APPROVED."""
        reviews = [
            _make_review("alice", "CHANGES_REQUESTED"),
            _make_review("alice", "APPROVED"),  # later review overrides
        ]
        with self._patch_paginate(reviews):
            result = _fetch_reviews(42, token=None)
        assert result["decision"] == "APPROVED"
        assert "alice" in result["approved_by"]
        assert result["changes_by"] == []

    def test_pending_when_no_reviews(self):
        with self._patch_paginate([]):
            result = _fetch_reviews(42, token=None)
        assert result["decision"] == "PENDING"
        assert result["review_count"] == 0

    def test_commented_only(self):
        with self._patch_paginate([_make_review("carol", "COMMENTED")]):
            result = _fetch_reviews(42, token=None)
        assert result["decision"] == "COMMENTED"

    def test_api_error_returns_unknown(self):
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent._paginate",
            side_effect=RuntimeError("rate limit"),
        ):
            result = _fetch_reviews(42, token=None)
        assert result["decision"] == "unknown"

    def test_dismissed_not_counted(self):
        """Dismissed reviews should not appear in approved_by or changes_by."""
        with self._patch_paginate([_make_review("dave", "DISMISSED")]):
            result = _fetch_reviews(42, token=None)
        assert result["approved_by"] == []
        assert result["changes_by"] == []


# ── _fetch_check_status ───────────────────────────────────────────────────────


class TestFetchCheckStatus:
    def _pr(self, sha="abc123"):
        return {"head": {"sha": sha}}

    def test_success(self):
        data = {"check_runs": [_make_check_run("completed", "success")]}
        with patch("ocs_ci.ocs.dr.dr_github_agent._get", return_value=data):
            assert (
                _fetch_check_status(self._pr(), token=None, fetch_checks=True)
                == "success"
            )

    def test_failure(self):
        data = {"check_runs": [_make_check_run("completed", "failure")]}
        with patch("ocs_ci.ocs.dr.dr_github_agent._get", return_value=data):
            assert (
                _fetch_check_status(self._pr(), token=None, fetch_checks=True)
                == "failure"
            )

    def test_pending_when_in_progress(self):
        data = {"check_runs": [_make_check_run("in_progress", None)]}
        with patch("ocs_ci.ocs.dr.dr_github_agent._get", return_value=data):
            assert (
                _fetch_check_status(self._pr(), token=None, fetch_checks=True)
                == "pending"
            )

    def test_pending_when_no_runs(self):
        data = {"check_runs": []}
        with patch("ocs_ci.ocs.dr.dr_github_agent._get", return_value=data):
            assert (
                _fetch_check_status(self._pr(), token=None, fetch_checks=True)
                == "pending"
            )

    def test_skipped_when_fetch_checks_false(self):
        assert (
            _fetch_check_status(self._pr(), token=None, fetch_checks=False) == "skipped"
        )

    def test_unknown_when_no_sha(self):
        assert (
            _fetch_check_status({"head": {"sha": ""}}, token=None, fetch_checks=True)
            == "unknown"
        )

    def test_api_error_returns_unknown(self):
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent._get", side_effect=RuntimeError("err")
        ):
            assert (
                _fetch_check_status(self._pr(), token=None, fetch_checks=True)
                == "unknown"
            )

    def test_timed_out_is_failure(self):
        data = {"check_runs": [_make_check_run("completed", "timed_out")]}
        with patch("ocs_ci.ocs.dr.dr_github_agent._get", return_value=data):
            assert (
                _fetch_check_status(self._pr(), token=None, fetch_checks=True)
                == "failure"
            )


# ── _fetch_files_changed ──────────────────────────────────────────────────────


class TestFetchFilesChanged:
    def test_returns_count(self):
        files = [{"filename": "a.py"}, {"filename": "b.py"}]
        with patch("ocs_ci.ocs.dr.dr_github_agent._paginate", return_value=files):
            assert _fetch_files_changed(99, token=None) == 2

    def test_error_returns_minus_one(self):
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent._paginate", side_effect=Exception("err")
        ):
            assert _fetch_files_changed(99, token=None) == -1


# ── _enrich_pr ────────────────────────────────────────────────────────────────


class TestEnrichPr:
    def test_basic_shape(self):
        pr = _make_pr(
            number=500,
            title="rdr: add failover test",
            labels=["rdr", "test"],
            assignees=["alice"],
            requested_reviewers=["bob"],
            milestone="RDR-4.17",
        )
        with (
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_reviews",
                return_value={
                    "decision": "APPROVED",
                    "approved_by": ["alice"],
                    "changes_by": [],
                    "review_count": 1,
                },
            ),
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_check_status",
                return_value="success",
            ),
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_files_changed",
                return_value=4,
            ),
        ):
            result = _enrich_pr(pr, token=None, fetch_checks=True)

        assert result["number"] == 500
        assert result["title"] == "rdr: add failover test"
        assert result["status"] == "open"
        assert result["labels"] == ["rdr", "test"]
        assert result["assignees"] == ["alice"]
        assert result["requested_reviewers"] == ["bob"]
        assert result["milestone"] == "RDR-4.17"
        assert result["review_decision"] == "APPROVED"
        assert result["approved_by"] == ["alice"]
        assert result["ci_status"] == "success"
        assert result["files_changed"] == 4
        assert result["comments"] == 3  # 2 + 1 from fixture

    def test_merged_pr_status(self):
        pr = _make_pr(state="closed", merged_at="2024-03-01T10:00:00Z")
        with (
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_reviews",
                return_value={
                    "decision": "APPROVED",
                    "approved_by": [],
                    "changes_by": [],
                    "review_count": 0,
                },
            ),
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_check_status",
                return_value="success",
            ),
            patch("ocs_ci.ocs.dr.dr_github_agent._fetch_files_changed", return_value=0),
        ):
            result = _enrich_pr(pr, token=None, fetch_checks=False)
        assert result["status"] == "merged"
        assert result["merged_at"] == "2024-03-01T10:00:00Z"

    def test_draft_pr_status(self):
        pr = _make_pr(draft=True)
        with (
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_reviews",
                return_value={
                    "decision": "PENDING",
                    "approved_by": [],
                    "changes_by": [],
                    "review_count": 0,
                },
            ),
            patch(
                "ocs_ci.ocs.dr.dr_github_agent._fetch_check_status",
                return_value="pending",
            ),
            patch("ocs_ci.ocs.dr.dr_github_agent._fetch_files_changed", return_value=1),
        ):
            result = _enrich_pr(pr, token=None, fetch_checks=True)
        assert result["status"] == "draft"
        assert result["draft"] is True


# ── DRGitHubAgent.list_rdr_prs ────────────────────────────────────────────────


class TestDRGitHubAgentListRdrPrs:
    def _make_agent(self):
        return DRGitHubAgent(token="fake-token", fetch_checks=False)

    def test_filters_and_enriches(self):
        raw_prs = [
            _make_pr(number=1, title="rdr: failover test"),
            _make_pr(
                number=2, title="fix: unrelated mcg bug", head_ref="mcg/fix", labels=[]
            ),
            _make_pr(number=3, title="regional dr topology"),
        ]
        enriched_pr = {
            "number": 0,
            "title": "",
            "url": "",
            "status": "open",
            "draft": False,
            "author": "dev",
            "created_at": "",
            "updated_at": "",
            "merged_at": None,
            "age_days": 0,
            "labels": [],
            "assignees": [],
            "requested_reviewers": [],
            "milestone": None,
            "review_decision": "PENDING",
            "approved_by": [],
            "changes_requested_by": [],
            "review_count": 0,
            "comments": 0,
            "files_changed": 0,
            "ci_status": "skipped",
            "base_branch": "master",
            "head_branch": "rdr/fix",
        }

        def fake_enrich(pr, token, fetch_checks):
            return {**enriched_pr, "number": pr["number"], "title": pr["title"]}

        agent = self._make_agent()
        with (
            patch.object(agent, "_fetch_all_prs", return_value=raw_prs),
            patch("ocs_ci.ocs.dr.dr_github_agent._enrich_pr", side_effect=fake_enrich),
        ):
            result = agent.list_rdr_prs(state="open")

        # PR #2 (unrelated) must be excluded
        numbers = [p["number"] for p in result]
        assert 1 in numbers
        assert 3 in numbers
        assert 2 not in numbers

    def test_empty_when_no_rdr_prs(self):
        raw_prs = [
            _make_pr(
                number=10, title="fix: mcg bucket", head_ref="fix/mcg", labels=["mcg"]
            ),
        ]
        agent = self._make_agent()
        with patch.object(agent, "_fetch_all_prs", return_value=raw_prs):
            result = agent.list_rdr_prs()
        assert result == []

    def test_sorted_descending_by_number(self):
        raw_prs = [
            _make_pr(number=10, title="rdr fix one"),
            _make_pr(number=30, title="rdr fix two"),
            _make_pr(number=20, title="rdr fix three"),
        ]
        enriched_base = {
            "title": "",
            "url": "",
            "status": "open",
            "draft": False,
            "author": "dev",
            "created_at": "",
            "updated_at": "",
            "merged_at": None,
            "age_days": 0,
            "labels": [],
            "assignees": [],
            "requested_reviewers": [],
            "milestone": None,
            "review_decision": "PENDING",
            "approved_by": [],
            "changes_requested_by": [],
            "review_count": 0,
            "comments": 0,
            "files_changed": 0,
            "ci_status": "skipped",
            "base_branch": "master",
            "head_branch": "rdr/x",
        }

        def fake_enrich(pr, token, fetch_checks):
            return {**enriched_base, "number": pr["number"]}

        agent = self._make_agent()
        with (
            patch.object(agent, "_fetch_all_prs", return_value=raw_prs),
            patch("ocs_ci.ocs.dr.dr_github_agent._enrich_pr", side_effect=fake_enrich),
        ):
            result = agent.list_rdr_prs()

        assert [p["number"] for p in result] == [30, 20, 10]


# ── Output smoke tests ────────────────────────────────────────────────────────


class TestPrintMethods:
    def _sample_enriched_pr(self, number: int = 101) -> dict:
        return {
            "number": number,
            "title": "rdr: add failover e2e test",
            "url": f"https://github.com/red-hat-storage/ocs-ci/pull/{number}",
            "status": "open",
            "draft": False,
            "author": "dev-user",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-05T00:00:00Z",
            "merged_at": None,
            "age_days": 30,
            "labels": ["rdr", "test"],
            "assignees": ["alice"],
            "requested_reviewers": ["bob"],
            "milestone": "RDR-4.17",
            "review_decision": "APPROVED",
            "approved_by": ["alice"],
            "changes_requested_by": [],
            "review_count": 1,
            "comments": 3,
            "files_changed": 5,
            "ci_status": "success",
            "base_branch": "master",
            "head_branch": "rdr/failover-fix",
        }

    def test_print_table_does_not_crash(self, capsys):
        DRGitHubAgent.print_table([self._sample_enriched_pr()])
        out = capsys.readouterr().out
        assert "101" in out
        assert "rdr: add failover" in out

    def test_print_table_empty(self, capsys):
        DRGitHubAgent.print_table([])
        out = capsys.readouterr().out
        assert "No RDR" in out

    def test_print_summary_does_not_crash(self, capsys):
        DRGitHubAgent.print_summary(
            [self._sample_enriched_pr(101), self._sample_enriched_pr(102)]
        )
        out = capsys.readouterr().out
        assert "Total RDR PRs" in out
        assert "2" in out

    def test_print_detail_does_not_crash(self, capsys):
        DRGitHubAgent.print_detail(self._sample_enriched_pr())
        out = capsys.readouterr().out
        assert "PR #101" in out
        assert "APPROVED" in out
        assert "alice" in out


# ── _resolve_token ────────────────────────────────────────────────────────────


class TestResolveToken:
    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "env-token-xyz")
        assert _resolve_token() == "env-token-xyz"

    def test_from_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        token_file = tmp_path / ".github_token"
        token_file.write_text("file-token-abc\n")
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.os.path.expanduser",
            return_value=str(token_file),
        ):
            result = _resolve_token()
        assert result == "file-token-abc"

    def test_returns_none_when_nothing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        nonexistent = str(tmp_path / "no_such_file")
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.os.path.expanduser", return_value=nonexistent
        ):
            result = _resolve_token()
        assert result is None

    def test_env_var_takes_precedence(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GITHUB_TOKEN", "env-wins")
        token_file = tmp_path / ".github_token"
        token_file.write_text("file-token\n")
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.os.path.expanduser",
            return_value=str(token_file),
        ):
            result = _resolve_token()
        assert result == "env-wins"


# ── export_html ───────────────────────────────────────────────────────────────


class TestExportHtml:
    """Tests for DRGitHubAgent.export_html()."""

    def _sample_prs(self) -> list:
        return [
            {
                "number": 101,
                "title": "rdr: add failover e2e <test> & 'more'",
                "url": "https://github.com/red-hat-storage/ocs-ci/pull/101",
                "status": "open",
                "draft": False,
                "author": "alice",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-05T00:00:00Z",
                "merged_at": None,
                "age_days": 10,
                "labels": ["rdr", "Squad/Turquoise"],
                "assignees": ["alice"],
                "requested_reviewers": ["bob"],
                "milestone": "RDR-4.17",
                "review_decision": "APPROVED",
                "approved_by": ["bob"],
                "changes_requested_by": [],
                "review_count": 1,
                "comments": 3,
                "files_changed": 5,
                "ci_status": "success",
                "base_branch": "master",
                "head_branch": "rdr/failover",
            },
            {
                "number": 99,
                "title": "regional dr: changes requested PR",
                "url": "https://github.com/red-hat-storage/ocs-ci/pull/99",
                "status": "open",
                "draft": True,
                "author": "carol",
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-06T00:00:00Z",
                "merged_at": None,
                "age_days": 5,
                "labels": ["rdr"],
                "assignees": [],
                "requested_reviewers": [],
                "milestone": None,
                "review_decision": "CHANGES_REQUESTED",
                "approved_by": [],
                "changes_requested_by": ["dave"],
                "review_count": 1,
                "comments": 1,
                "files_changed": 2,
                "ci_status": "failure",
                "base_branch": "master",
                "head_branch": "rdr/changes",
            },
        ]

    def test_creates_file(self, tmp_path):
        out = tmp_path / "report.html"
        result = DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        assert result == str(out)
        assert out.exists()

    def test_html_contains_pr_numbers(self, tmp_path):
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "#101" in content
        assert "#99" in content

    def test_html_contains_pr_urls(self, tmp_path):
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "https://github.com/red-hat-storage/ocs-ci/pull/101" in content

    def test_html_escapes_special_chars_in_title(self, tmp_path):
        """Title with <, >, & must be HTML-escaped, not injected raw."""
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        # Raw < must not appear inside the title cell (it would break the table)
        assert "&lt;test&gt;" in content
        assert "&amp;" in content

    def test_html_contains_summary_cards(self, tmp_path):
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "Total PRs" in content
        assert "Approved" in content

    def test_html_contains_status_badges(self, tmp_path):
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "APPROVED" in content
        assert "CHANGES_REQUESTED" in content
        assert "success" in content
        assert "failure" in content

    def test_html_is_valid_structure(self, tmp_path):
        """File must open and close with correct HTML boilerplate."""
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content
        assert "<table" in content
        assert "</table>" in content

    def test_html_default_filename(self, tmp_path, monkeypatch):
        """Default filename is rdr_prs.html."""
        monkeypatch.chdir(tmp_path)
        result = DRGitHubAgent.export_html(self._sample_prs())
        assert result == "rdr_prs.html"
        assert (tmp_path / "rdr_prs.html").exists()

    def test_html_empty_prs(self, tmp_path):
        """Empty PR list must still produce a valid HTML file."""
        out = tmp_path / "empty.html"
        DRGitHubAgent.export_html([], path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "0 PR(s) matched" in content

    def test_html_label_pills_rendered(self, tmp_path):
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "Squad/Turquoise" in content

    def test_html_contains_sort_script(self, tmp_path):
        out = tmp_path / "report.html"
        DRGitHubAgent.export_html(self._sample_prs(), path=str(out))
        content = out.read_text(encoding="utf-8")
        assert "sortTable" in content
        assert "filterTable" in content


# ── post_slack ────────────────────────────────────────────────────────────────


class TestPostSlack:
    """Tests for DRGitHubAgent.post_slack()."""

    def _sample_prs(self) -> list:
        return [
            {
                "number": 200,
                "title": "rdr: failover test",
                "url": "https://github.com/red-hat-storage/ocs-ci/pull/200",
                "status": "open",
                "draft": False,
                "author": "eve",
                "created_at": "2024-02-01T00:00:00Z",
                "updated_at": "2024-02-05T00:00:00Z",
                "merged_at": None,
                "age_days": 8,
                "labels": ["rdr", "Squad/Turquoise"],
                "assignees": ["eve"],
                "requested_reviewers": [],
                "milestone": None,
                "review_decision": "PENDING",
                "approved_by": [],
                "changes_requested_by": [],
                "review_count": 0,
                "comments": 0,
                "files_changed": 3,
                "ci_status": "pending",
                "base_branch": "master",
                "head_branch": "rdr/test",
            }
        ]

    def _mock_slack_response(self, status_code: int = 200, text: str = "ok"):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        return mock_resp

    def test_returns_true_on_success(self):
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post",
            return_value=self._mock_slack_response(200),
        ):
            result = DRGitHubAgent.post_slack(
                self._sample_prs(), webhook_url="https://hooks.slack.com/fake"
            )
        assert result is True

    def test_returns_false_on_http_error(self):
        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post",
            return_value=self._mock_slack_response(400, "invalid_payload"),
        ):
            result = DRGitHubAgent.post_slack(
                self._sample_prs(), webhook_url="https://hooks.slack.com/fake"
            )
        assert result is False

    def test_payload_contains_pr_url(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["payload"] = kwargs.get("json")
            return self._mock_slack_response(200)

        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post", side_effect=fake_post
        ):
            DRGitHubAgent.post_slack(
                self._sample_prs(), webhook_url="https://hooks.slack.com/fake"
            )

        payload_str = str(captured["payload"])
        assert "pull/200" in payload_str

    def test_payload_has_blocks(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["payload"] = kwargs.get("json")
            return self._mock_slack_response(200)

        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post", side_effect=fake_post
        ):
            DRGitHubAgent.post_slack(
                self._sample_prs(), webhook_url="https://hooks.slack.com/fake"
            )

        assert "blocks" in captured["payload"]
        assert len(captured["payload"]["blocks"]) >= 3  # header + meta + divider

    def test_max_prs_cap(self):
        """When PRs > max_prs, a 'more PRs' context block must be appended."""
        big_list = self._sample_prs() * 5  # 5 PRs
        captured = {}

        def fake_post(url, **kwargs):
            captured["payload"] = kwargs.get("json")
            return self._mock_slack_response(200)

        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post", side_effect=fake_post
        ):
            DRGitHubAgent.post_slack(
                big_list, webhook_url="https://hooks.slack.com/fake", max_prs=2
            )

        blocks_str = str(captured["payload"]["blocks"])
        assert "more PRs not shown" in blocks_str

    def test_no_overflow_block_when_within_limit(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["payload"] = kwargs.get("json")
            return self._mock_slack_response(200)

        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post", side_effect=fake_post
        ):
            DRGitHubAgent.post_slack(
                self._sample_prs(),
                webhook_url="https://hooks.slack.com/fake",
                max_prs=20,
            )

        blocks_str = str(captured["payload"]["blocks"])
        assert "more PRs not shown" not in blocks_str

    def test_post_uses_correct_webhook_url(self):
        webhook = "https://hooks.slack.com/services/T123/B456/xyz"
        called_with = {}

        def fake_post(url, **kwargs):
            called_with["url"] = url
            return self._mock_slack_response(200)

        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post", side_effect=fake_post
        ):
            DRGitHubAgent.post_slack(self._sample_prs(), webhook_url=webhook)

        assert called_with["url"] == webhook

    def test_empty_prs_still_posts(self):
        """Empty list must still send a valid (zero-PR) Slack message."""
        captured = {}

        def fake_post(url, **kwargs):
            captured["payload"] = kwargs.get("json")
            return self._mock_slack_response(200)

        with patch(
            "ocs_ci.ocs.dr.dr_github_agent.requests.post", side_effect=fake_post
        ):
            result = DRGitHubAgent.post_slack(
                [], webhook_url="https://hooks.slack.com/fake"
            )

        assert result is True
        assert "blocks" in captured["payload"]
