from typing import *
from warnings import warn
from pandasql import sqldf
from scipy import stats
from rdkit import Chem
from tabulate import tabulate
from .utils import *
from tqdm import notebook

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import string


class polymers:
    def __init__(
        self,
        csvfile: str = None,
        database: str = None,
        table: str = None,
        printTable: bool = True,
    ) -> None:

        self.data = None
        self.sample = None
        self.props = None
        self.prop_ranges = None
        self.unique_monomers = None
        self.all_monomers = None
        self.printTable = printTable

        self._import(csvfile, database, table)

    def query(
        self,
        properties: List[Tuple[str, Tuple[float, float]]],
        # iterator: Tuple[any, any] = None,
    ) -> float:
        """Define the range or ranges of properties in the data, and get a subpopulation
        of the data that meets the set criteria.

        Args:
        -----
        properties List[Tuple[str, Tuple[float, float]]]:
            A list of the properties and
            their respective ranges:
                [
                    ('property_1'(lower_bound, upper_bound)),
                    ('property_2'(lower_bound, upper_bound)),
                ]
            etc...

        returns:
        --------
            The relative sample size as a percentage of the total data (float)
        """

        props = []
        prop_ranges = []
        for prop in properties:
            props.append(prop[0])
            prop_ranges.append(prop[1])
        self.prop_ranges = prop_ranges
        self.props = props

        self._get_sample()
        relSize = 100.0 * (len(self.sample) / len(self.data))
        return relSize

    def zscores(self) -> None:
        sample = self.sample
        totals = [self.all_monomers.count(x) for x in self.unique_monomers]
        scores = []
        zscores = pd.DataFrame(self.unique_monomers, columns=["monomer"])
        i = 0

        for frag_id in notebook.tqdm(
            self.unique_monomers,
            desc="Computing fragment z-scores",
        ):
            scores.append(self._compute_frag_zscore(frag_id, sample, totals[i]))
            i += 1
        zscores["zscore"] = scores
        self.zscores = zscores

    def display(
        self,
        plot: bool = True,
        table: bool = True,
        table_style: str = "github",
        k: int = 12,
    ) -> None:
        print(
            "\n" + tabulate(self.zscores, headers="keys", tablefmt=table_style) + "\n"
        )

    def _import(self, csv, db, tab) -> None:
        """Import the data from either a specified CSV file of from a table in a
        sqlite3 database; the CSV will have priority.
        Check the data for ascii_uppercase column names.
        """
        if csv:
            if db:
                warn(
                    "WARNING: You have selected both a .csv and sqlite database. The database will be ignored."
                )
                db = None
                tab = None
            data = pd.read_csv(csv)
        elif db:
            if not tab:
                raise Exception(f"You have not specified a table in the database {db}")
            conn = sqlite3.connect(db)
            tab = sanitizeSQL(tab)
            sql = f"SELECT * FROM {tab};"
            data = pd.read_sql_query(sql, conn)
            conn.commit()
            conn.close()
        else:
            raise Exception("You must provide a datasource!")
        m_cols = [i for i in data.columns if i in list(string.ascii_uppercase)]
        if len(m_cols) == 0:
            raise Exception(
                f"Data must have at least one monomer column with a name in {list(string.ascii_uppercase)}"
            )
        monomers = []
        for col in m_cols:
            monomers.append(data[col].to_list())

        self.all_monomers = [
            brushTeeth(m) for m in [s for col in monomers for s in col]
        ]
        self.unique_monomers = list(set(self.all_monomers))

        data.insert(0, "ID", range(0, len(data)))
        self.data = data

    def _get_sample(self) -> None:
        """
        Querying the data and returns a sample that meets specificied criteria

        Props and prop_ranges are converted into a SQL query and executed on a temporary
        dataframe, which lacks the fingerprint bit. The sample from the temporary df is
        then joined with the main data, using the ID field in order to get the fp bits back.

        We need a temporary classless df to query from, which is deleted later. For large
        datasets, this could cause memory issues, so again, might be best to use a sqlite db
        and query that directly?
        """

        params_list = []
        tmp_df = self.data[["ID"] + list(self.props)].copy()

        for i, prop in enumerate(self.props):
            params_list.append(
                f"{prop} >= {self.prop_ranges[i][0]} AND {prop} <= {self.prop_ranges[i][1]}"
            )
        params = " AND ".join(params_list)
        sql = "SELECT * FROM tmp_df WHERE " + params + ";"

        # get sample in specified property range
        queried = sqldf(sql)
        data = self.data
        sample = queried.merge(
            data,
            how="left",
            on="ID",
            suffixes=("", "__y__"),
        )
        sample.drop(
            sample.filter(regex="__y__$").columns.tolist(),
            axis=1,
            inplace=True,
        )
        del tmp_df
        self.sample = sample

    def _compute_frag_zscore(
        self, frag_id: Union[str, int], subpop: pd.DataFrame, total: int
    ) -> float:
        """Compute zscores for a given fragment.

        Args:
        -----
            frag_id (Union[str, int]): Fragment id. Either smiles string if user
            defined or integer of morgan fingerprint bit position if auto-generated.
            subpoop (DataFrame): Sample of population in specified property range.
            total (int): Total in population with fragment.

        Returns:
        --------
            float: Fragment zscore.
        """
        pop_total = total
        # selection_total = subpop[frag_id].sum()

        m_cols = [i for i in subpop.columns if i in list(string.ascii_uppercase)]
        monomers = []
        for col in m_cols:
            monomers.append(subpop[col].to_list())
        all_subpop = [brushTeeth(m) for m in [s for col in monomers for s in col]]
        selection_total = all_subpop.count(frag_id)

        N = len(self.data)  # total in population
        n = len(subpop)  # total in selection
        k = pop_total  # num in population with fragment
        x = selection_total  # num in selection with fragment

        # Using sp just so it's easy to switch functions if need be. Granted it's a little
        # slower than previous but easy enough to switch back
        use_scipy = False
        if use_scipy:
            mean = stats.hypergeom.mean(N, n, k)
            var = stats.hypergeom.var(N, n, k) + 1e-30
        else:
            mean = n * k / N
            var = n * k * (N - k) * (N - n) / (N ** 2 * (N - 1)) + 1e-30
        z = (x - mean) / var
        return z