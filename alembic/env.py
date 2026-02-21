"""
Alembic environment — wired to SystemConfig for DB connection
and Studio models for autogenerate support.

Only manages Studio tables (prompts, prompt_versions, workflows, etc.).
Legacy tables (document_ledger, metric_facts) remain on create_all().
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, create_engine
from alembic import context

from src.config import SystemConfig
from src.storage.studio_models import Base as StudioBase  # Import to register models

# Alembic Config object
config = context.config

# Set up loggers from .ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Build DB URL from SystemConfig (same connection as analytics_driver)
DATABASE_URL = (
    f"postgresql://{SystemConfig.POSTGRES_USER}:{SystemConfig.POSTGRES_PASSWORD}"
    f"@{SystemConfig.POSTGRES_HOST}:{SystemConfig.POSTGRES_PORT}"
    f"/{SystemConfig.POSTGRES_DB}"
)

# Point Alembic at the Studio models metadata
target_metadata = StudioBase.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (SQL script output, no DB connection)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (live DB connection)."""
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
