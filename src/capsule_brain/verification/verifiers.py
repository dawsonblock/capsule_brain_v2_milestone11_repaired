from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod
from typing import Any

from .models import VerificationCheck, VerificationStatus


class Verifier(ABC):
    name = "base"

    @abstractmethod
    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        raise NotImplementedError


class NonEmptyVerifier(Verifier):
    name = "non_empty"

    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        ok = bool(subject.strip())
        return VerificationCheck(
            verifier=self.name,
            status=(
                VerificationStatus.PASS
                if ok
                else VerificationStatus.FAIL
            ),
            summary=(
                "Subject is non-empty."
                if ok
                else "Subject is empty."
            ),
        )


class PythonSyntaxVerifier(Verifier):
    name = "python_syntax"

    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        if metadata.get("content_type") != "python":
            return VerificationCheck(
                self.name,
                VerificationStatus.SKIP,
                "Not Python.",
            )

        try:
            ast.parse(subject)
            return VerificationCheck(
                self.name,
                VerificationStatus.PASS,
                "Python syntax is valid.",
            )
        except SyntaxError as exc:
            return VerificationCheck(
                self.name,
                VerificationStatus.FAIL,
                "Python syntax is invalid.",
                {
                    "line": exc.lineno,
                    "message": exc.msg,
                },
            )


class JSONSyntaxVerifier(Verifier):
    name = "json_syntax"

    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        if metadata.get("content_type") != "json":
            return VerificationCheck(
                self.name,
                VerificationStatus.SKIP,
                "Not JSON.",
            )

        try:
            json.loads(subject)
            return VerificationCheck(
                self.name,
                VerificationStatus.PASS,
                "JSON syntax is valid.",
            )
        except json.JSONDecodeError as exc:
            return VerificationCheck(
                self.name,
                VerificationStatus.FAIL,
                "JSON syntax is invalid.",
                {
                    "line": exc.lineno,
                    "column": exc.colno,
                },
            )


class RequiredFieldsVerifier(Verifier):
    name = "required_fields"

    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        required = list(
            metadata.get("required_fields") or []
        )
        if not required:
            return VerificationCheck(
                self.name,
                VerificationStatus.SKIP,
                "No required fields.",
            )

        try:
            obj = json.loads(subject)
        except Exception as exc:
            # Fail-closed: if the subject is not valid JSON, the required
            # fields check cannot pass. Returning SKIP would let malformed
            # output silently bypass field validation.
            return VerificationCheck(
                self.name,
                VerificationStatus.FAIL,
                "Subject is not valid JSON; required field check failed.",
                {"error": str(exc)},
            )

        if not isinstance(obj, dict):
            return VerificationCheck(
                self.name,
                VerificationStatus.FAIL,
                "JSON subject is not an object.",
                {"expected": "object", "got": type(obj).__name__},
            )

        missing = [key for key in required if key not in obj]
        return VerificationCheck(
            self.name,
            (
                VerificationStatus.FAIL
                if missing
                else VerificationStatus.PASS
            ),
            (
                "Missing required fields."
                if missing
                else "Required fields present."
            ),
            {"missing": missing},
        )


class ConsistencyVerifier(Verifier):
    name = "consistency"

    async def verify(
        self,
        subject: str,
        metadata: dict[str, Any],
    ) -> VerificationCheck:
        expected = (
            metadata.get("expected_contains") or []
        )
        forbidden = (
            metadata.get("forbidden_contains") or []
        )

        if not expected and not forbidden:
            return VerificationCheck(
                self.name,
                VerificationStatus.SKIP,
                "No constraints.",
            )

        problems = [
            f"missing:{value}"
            for value in expected
            if value not in subject
        ]
        problems.extend(
            f"forbidden:{value}"
            for value in forbidden
            if value in subject
        )

        return VerificationCheck(
            self.name,
            (
                VerificationStatus.FAIL
                if problems
                else VerificationStatus.PASS
            ),
            (
                "Consistency failed."
                if problems
                else "Consistency passed."
            ),
            {"problems": problems},
        )
