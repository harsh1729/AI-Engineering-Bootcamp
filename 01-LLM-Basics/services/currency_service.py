import requests


class CurrencyService:

    BASE_URL = "https://api.frankfurter.app/latest"

    def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> str:

        response = requests.get(
            self.BASE_URL,
            params={
                "amount": amount,
                "from": from_currency.upper(),
                "to": to_currency.upper(),
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        converted_amount = data["rates"][to_currency.upper()]

        return (
            f"{amount:.2f} {from_currency.upper()} = "
            f"{converted_amount:.2f} {to_currency.upper()}"
        )