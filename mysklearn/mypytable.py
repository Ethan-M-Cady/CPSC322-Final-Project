"""
This module is python representation of a table(2D list) structure

Classes:
    MyPyTable: 2D table of data with column names
"""
import copy
import csv
from math import ceil, floor
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True, narep="NA"):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        try:
            index = self.column_names.index(col_identifier)
        except Exception as exc:
            raise ValueError(f"Invalid Column Name: {col_identifier}") from exc
        col = []
        if include_missing_values:
            for row in self.data:
                col.append(row[index])
        else:
            for row in self.data:
                if row[index] != narep:
                    col.append(row[index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for item in row:
                try:
                    row[row.index(item)] = float(item)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_table = []
        for row in self.data:
            if self.data.index(row) not in row_indexes_to_drop:
                new_table.append(row)
        self.data = new_table

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        with open(filename) as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                table.append(row)
            self.column_names = table.pop(0)
            self.data = copy.deepcopy(table)
            self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        known_key_pairs = []
        key_index = []
        dups = []
        for key in key_column_names:
            key_index.append(self.column_names.index(key))
        for row in self.data:
            key_pair = []
            for k_index in key_index:
                key_pair.append(row[k_index])
            if key_pair in known_key_pairs:
                dups.append(self.data.index(row))
            else:
                known_key_pairs.append(key_pair)
        return dups

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_table = []
        for row in self.data:
            if "NA" not in row:
                new_table.append(row)
        self.data = copy.deepcopy(new_table)

    def remove_rows_with_missing_values_given_col(self, col_identifier, narep="NA"):
        """Remove rows from the table data that contain a missing value ("NA") based on a specific column.
        """
        index = self.column_names.index(col_identifier)
        new_table = []
        for row in self.data:
            if narep is not row[index]:
                new_table.append(row)
        self.data = copy.deepcopy(new_table)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        index = self.column_names.index(col_name)
        col = self.get_column(col_name, include_missing_values=False)
        average = sum(col) / len(col)
        for row in self.data:
            if row[index] == "NA":
                row[index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_names = ["attribute", "min", "max", "mid", "avg", "median"]
        data = []
        for name in col_names:
            col_data = []
            col = self.get_column(name, include_missing_values=False)
            if len(col) != 0:
                for item in stats_names:
                    if item == "attribute":
                        col_data.append(name)
                    elif item == "min":
                        col_data.append(min(col))
                    elif item == "max":
                        col_data.append(max(col))
                    elif item == "mid":
                        col_data.append((max(col)+min(col))/2)
                    elif item == "avg":
                        col_data.append(sum(col)/ len(col))
                    elif item == "median":
                        if len(col) % 2 == 1:
                            col_data.append(sorted(col)[floor(len(col)/2)])
                        else:
                            index = ceil(len(col)/2)
                            median = (sorted(col)[index-1]+sorted(col)[index]) / 2
                            col_data.append(median)
                data.append(col_data)
        return MyPyTable(column_names=stats_names, data=data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        this_key_index = self.make_key_index(other_table, key_column_names)[0]
        other_key_index = self.make_key_index(other_table, key_column_names)[1]
        new_header = self.make_new_header(other_table)
        new_data = []
        for row in self.data:
            for o_row in other_table.data:
                this_key_pair = [row[k_index] for k_index in this_key_index]
                other_key_pair = [o_row[k_index] for k_index in other_key_index]
                if this_key_pair == other_key_pair:
                    new_row = copy.deepcopy(row)
                    for item in o_row:
                        if item not in other_key_pair:
                            new_row.append(item)
                    new_data.append(new_row)
        return MyPyTable(new_header, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        inner_table = self.perform_inner_join(other_table, key_column_names).data
        this_key_index = self.make_key_index(other_table, key_column_names)[0]
        other_key_index = self.make_key_index(other_table, key_column_names)[1]
        new_header = self.make_new_header(other_table)
        for row in self.data:
            r_pair = []
            for r_row in inner_table:
                r_pair.append([r_row[new_header.index(k_index)] for k_index in key_column_names])
            this_key_pair = [row[k_index] for k_index in this_key_index]
            if this_key_pair not in r_pair:
                new_row = []
                for col_name in new_header:
                    if col_name not in self.column_names:
                        new_row.append("NA")
                    else:
                        new_row.append(row[self.column_names.index(col_name)])
                inner_table.append(new_row)
        for o_row in other_table.data:
            r_pair = []
            for r_row in inner_table:
                r_pair.append([r_row[new_header.index(k_index)] for k_index in key_column_names])
            other_key_pair = [o_row[k_index] for k_index in other_key_index]
            if other_key_pair not in r_pair:
                new_row = []
                for col_name in new_header:
                    if col_name not in other_table.column_names:
                        new_row.append("NA")
                    else:
                        new_row.append(o_row[other_table.column_names.index(col_name)])
                inner_table.append(new_row)
        return MyPyTable(new_header, inner_table)

    def make_new_header(self, other_table):
        """Return a new list that is the new MyPyTable's header row

        Args:
            other_table(MyPyTable): the second table to join with this table's header.

        Returns:
            new_header: the new header row
        """
        new_header = copy.deepcopy(self.column_names)
        for col_name in other_table.column_names:
            if col_name not in new_header:
                new_header.append(col_name)
        return new_header

    def make_key_index(self, other_table, key_column_names):
        """Return two new list that is the corresponding key column indexes for this table
           and other table

        Args:
            other_table(MyPyTable): the second table to join with this table's header.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            this_key_index: corresponding key index for this table
            other_key_index: corresponding key index for other_table
        """
        this_key_index = []
        other_key_index = []
        for key_col in key_column_names:
            this_key_index.append(self.column_names.index(key_col))
            other_key_index.append(other_table.column_names.index(key_col))
        return this_key_index, other_key_index