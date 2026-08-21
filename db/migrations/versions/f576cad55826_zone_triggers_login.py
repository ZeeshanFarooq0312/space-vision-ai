"""zone triggers_login flag

Revision ID: f576cad55826
Revises: 53f6a45be0f5
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f576cad55826'
down_revision: Union[str, None] = '53f6a45be0f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'zones',
        sa.Column('triggers_login', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column('zones', 'triggers_login', server_default=None)


def downgrade() -> None:
    op.drop_column('zones', 'triggers_login')
