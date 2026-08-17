"""Create or promote an admin user in Postgres.

Usage (from backend/):
    python -m scripts.create_admin --name "Harsh" --email "you@example.com" --password "secret"

Public /auth/register always sets is_admin=False; use this script for admins only.
"""

import argparse
import asyncio
import sys

from app.database.database import AsyncSessionLocal
from app.database.repositories.user_repository import UserRepository
from app.services.password_service import hash_password


async def create_admin(*, name: str, email: str, password: str) -> None:
    normalized_email = email.strip().lower()

    async with AsyncSessionLocal() as session:
        repo = UserRepository(session)
        existing = await repo.get_by_email(normalized_email)

        if existing is not None:
            if existing.is_admin:
                print(f"Admin already exists: {normalized_email}")
                return

            existing.is_admin = True
            existing.is_approved = True
            existing.is_enabled = True
            existing.name = name.strip()
            existing.password_hash = hash_password(password)
            await session.commit()
            print(f"Promoted existing user to admin: {normalized_email}")
            return

        await repo.create(
            name=name.strip(),
            email=normalized_email,
            password_hash=hash_password(password),
            is_approved=True,
            is_admin=True,
            is_enabled=True,
        )
        await session.commit()
        print(f"Created admin user: {normalized_email}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or promote an admin user.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    try:
        asyncio.run(
            create_admin(
                name=args.name,
                email=args.email,
                password=args.password,
            )
        )
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
