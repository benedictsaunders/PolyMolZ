from typing import *
from unicodedata import normalize
from warnings import warn
import string
import itertools
import re
from numpy.lib.arraysetops import unique

from tabulate import tabulate
from .utils import *
from tqdm import notebook

from scipy import stats
import numpy as np
import pandas as pd
from pandasql import sqldf
import sqlite3

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cairosvg import svg2png

from rdkit import Chem


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
        self.zscores = None
        self.aliases = False

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

    # def iterative_query(
    #     self,
    #     iterator: Union[list, iter, np.array] = None,
    #     vars: Union[int, float, List[Union[int, float]]] = None,
    #     func: function = None,
    # ):
    #     def same(x):
    #         return x

    #     if not func:
    #         func = same

    #     for p in iterator:
    #         pass

    def getZscores(self, aliases="auto") -> None:
        """Calculation of zscores by iterating over the unique monomers

        Returns:
        --------
            scores
        """
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
        print(scores)
        self.zscores = zscores
        self._process_aliases(aliases=aliases)
        aliases = zscores.alias.to_list()
        return aliases, np.array(scores)

    def display(
        self,
        k: int = 8,
        plot: bool = True,
        log_y: bool = True,
        table: bool = True,
        top_only: bool = True,
        fragment_grid: bool = True,
        per_row: int = 8,
        aliases: Union[str, List[str]] = None,
        table_style: str = "github",
    ) -> None:
        """Display the zscore data as specified by the user, as a table plot or
        both with the top k values.
        In future, heatmap for zscores over a range of ranges/conditions?

        Args:
        -----
            k: specify how many zscores should be displayed, from the largest
            plot: specify if a plot of zscores should be displayed
            table: specify whether a table of zscores should be displayed
            aliases: list of aliases for the SMILES strings, to make the outputs
                easier to decipher. If provided, aliases will replace the tick
                labels on the plot, but are added as a third column in the table
                Can be 'auto', in which case they will be assigned names such as
                'A1', 'A2' etc.
            per_row: number of molecules drawn per row in fragment_grid. Also
                defines the naming scheme for auto aliases i.e. per_row = 8:
                A1 -> A8; B1 -> B8 etc.
            table_style: the style of table, from the `tabulate' module

        Returns:
        --------
            None: None

        """
        if not plot and not table:
            warn("No display style selected.")

        if aliases:
            self._process_aliases(aliases, per_row)
            aliases = self.zscores.alias.to_list()
            self.aliases = True

        if fragment_grid:
            mols = [Chem.MolFromSmiles(m) for m in list(self.unique_monomers)]
            img = Chem.Draw.MolsToGridImage(
                mols=mols,
                molsPerRow=per_row,
                useSVG=True,
                legends=aliases,
                maxMols=len(self.unique_monomers),
            )

            svg2png(bytestring=img.data, write_to="fragments.png")
        disp_zscores = self.zscores.sort_values(
            by=["zscore"],
            ascending=False,
            inplace=False,
        )
        scores = disp_zscores.zscore.to_list()
        if aliases:
            ids = disp_zscores.alias.to_list()
        else:
            ids = disp_zscores.monomer.to_list()

        if table:
            printable = disp_zscores
            print(
                "\n"
                + tabulate(
                    printable,
                    headers="keys",
                    tablefmt=table_style,
                    showindex=False,
                )
                + "\n"
            )
        data = dict(zip(ids, scores))
        if plot:
            self._show_plot(
                data=data,
                k=k,
                top_only=top_only,
                save=False,
                figsize=(8, 6),
                log_y=log_y,
            )

    def heatmap(self) -> None:
        zscores = self.zscores
        pass

    def _import(
        self,
        csv: str = None,
        db: str = None,
        tab: str = None,
    ) -> None:
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

        tot = list(self.all_monomers)
        unq = []
        for m in tot:
            if m not in unq:
                unq.append(m)

        self.unique_monomers = unq

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
        self,
        frag_id: Union[str, int],
        subpop: pd.DataFrame,
        total: int,
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

    def _process_aliases(
        self,
        aliases: Union[str, dict] = None,
        per_row: int = 8,
    ) -> None:
        if "alias" in self.zscores.columns:
            return None
        labels = []
        if type(aliases) is dict:
            labels = aliases
        elif aliases == "auto":
            letts = list(string.ascii_uppercase)
            length = len(self.zscores)
            if length > per_row * 26:
                letts = itertools.combinations_with_replacement(letts, 2)
            i, j, = (
                0,
                1,
            )
            for x in range(length):
                labels.append(f"{letts[i]}{str(j)}")
                j += 1
                if j > per_row:
                    i += 1
                    j = 1
        else:
            raise Exception("Aliases must be dict type or 'auto'.")
        if len(labels) != len(self.zscores):
            print("Houston, we have a problem")
        else:
            self.zscores.insert(1, "alias", labels)

    def _show_plot(
        self,
        data: dict,
        k: int = None,
        top_only: bool = True,
        save: Union[str, bool] = None,
        figsize: Tuple[int, int] = (8, 6),
        log_y: bool = False,
    ) -> None:
        # data = {v: u for v, u in sorted(data.items(), key=lambda x: x[1])}
        ids = list(data.keys())
        scores = list(data.values())
        if k:
            ids = ids[:k] + ids[-k:]
            scores = scores[:k] + scores[-k:]

        if top_only:
            ids = ids[:k]
            scores = scores[:k]

        scores = scores[::-1]
        ids = ids[::-1]

        color_map = cm.get_cmap("RdYlGn")
        normalization = Normalize(vmin=-max(scores), vmax=max(scores))
        clr = color_map(normalization(scores))

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.bar(
            ids,
            scores,
            color=clr,
            width=0.4,
            log=log_y,
        )
        ax.set_ylabel("z-score (std. dev.)")
        if not self.aliases:
            plt.xticks(rotation=90)
        plt.tight_layout()
        if save:
            plt.savefig(save)
        plt.show()
