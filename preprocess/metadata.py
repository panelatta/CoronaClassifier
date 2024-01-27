import re
from typing import Any
import pandas as pd


def extract_clade(
    cur_df: pd.DataFrame,
    row: Any,
    clade_encoder: dict[str, int],
    encode_counter: int
) -> tuple[dict[str, int], int]:
    """
    Extracts the clade from the Nextstrain_clade column.
    :param cur_df:
    :param row:
    :param clade_encoder:
    :param encode_counter:
    :return tuple[dict[str, int], int]:
    """

    match = re.search(r'\d+[A-Z]+', row.Nextstrain_clade)
    if match:
        if match.group() not in clade_encoder:
            clade_encoder[match.group()] = encode_counter
            encode_counter += 1
        cur_df.at[row.Index, 'Nextstrain_clade'] = clade_encoder[match.group()]

    return clade_encoder, encode_counter
