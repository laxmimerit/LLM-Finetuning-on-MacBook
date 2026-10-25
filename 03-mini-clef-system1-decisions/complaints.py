"""The task: route a consumer complaint to one of 9 financial products.

ROUTES are the allowed options, `request()` builds a Jev / SystemOne-style request body, and `chat_messages()` /
`parse()` are for asking a normal chat model the same question and reading its JSON reply.
"""

import json
import re

ROUTES = {
    "credit_reporting": "credit reports, credit bureaus, credit repair, identity on report",
    "debt_collection": "debt collectors, collection calls, debts owed",
    "credit_card": "credit cards and prepaid cards",
    "bank_account": "checking or savings accounts",
    "mortgage": "home loans and mortgages",
    "money_transfer": "money transfers, wire, payment apps, virtual currency",
    "vehicle_loan": "car loans or vehicle leases",
    "student_loan": "student loans",
    "personal_loan": "payday, title, personal or installment loans",
}


def request(complaint, routes=ROUTES):
    """A SystemOne request: the state, and one typed question with its allowed options."""
    return {
        "model": "mini-clef",
        "state": {"complaint": complaint},
        "questions": {
            "product": {
                "type": "choice",
                "instructions": "Which financial product is the consumer's `complaint` about?",
                "criteria": dict(routes),
            }
        },
    }


SYSTEM = (
    "You are a decision model. You get a JSON request with a `state` and typed `questions`. "
    "Answer every question with exactly one of its criteria keys. "
    'Reply with JSON only, like {"product": "<key>"}.'
)


def chat_messages(complaint):
    req = request(complaint)
    body = {"state": req["state"], "questions": req["questions"]}
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(body)}]


def parse(reply):
    """The product a chat model wrote, or None when its reply has no valid choice."""
    for m in reversed(re.findall(r"\{[^{}]*\}", reply.split("</think>")[-1])):
        try:
            product = json.loads(m).get("product")
        except Exception:
            continue
        return product if product in ROUTES else None
    return None
