from llm_sdk.services import CurrencyService

from llm_sdk.models.tools import LLMTool, LLMToolParam


currency_service = CurrencyService()


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> str:

    return currency_service.convert_currency(
        amount=amount,
        from_currency=from_currency,
        to_currency=to_currency,
    )

CONVERT_CURRENCY_TOOL = LLMTool(
    name="convert_currency",
    description="Converts an amount from one currency to another using the latest exchange rate.",
    parameters=[
        LLMToolParam(
            name="amount",
            type="number",
            description="Amount to convert.",
        ),
        LLMToolParam(
            name="from_currency",
            type="string",
            description="Three-letter source currency code such as USD, INR or EUR.",
        ),
        LLMToolParam(
            name="to_currency",
            type="string",
            description="Three-letter target currency code such as USD, INR or EUR.",
        ),
    ],
)