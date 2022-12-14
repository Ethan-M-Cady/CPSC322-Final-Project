from mysklearn import myutils

import copy
import csv
import os
from pickle import TRUE
from tabulate import tabulate
# from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests


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
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):

        N = len(self.data)  # gets length of rows
        M = len(self.data[0])  # gets length of columns

        return N, M  # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):

        got_column = []
        try:
            index = self.column_names.index(col_identifier)
        except ValueError:  # invalid col_identifier
            raise ValueError
        if include_missing_values == True:
            for row in self.data:
                got_column.append(row[index])
        else:
            for row in self.data:
                if row[index] != "NA":
                    got_column.append(row[index])
        return got_column

    def convert_to_numeric(self):
        for row in range(len(self.data)):
            for column in range(len(self.data[row])):
                try:
                    # converting every val to float
                    self.data[row][column] = float(self.data[row][column])
                except:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        row_indexes_to_drop.sort(reverse=True)
        for row in row_indexes_to_drop:
            self.data.pop(row)

    def load_from_file(self, filename):
        with open(filename, mode='r') as f:
            csv_f = csv.reader(f)
            for lines in csv_f:
                self.data.append(lines)
        self.convert_to_numeric()  # after load so calls respective function
        self.column_names = self.data[0]
        del (self.data[0])
        return self

    def save_to_file(self, filename):
        with open(filename, 'w') as CSV_file:
            CSV_writer = csv.writer(CSV_file)
            CSV_writer.writerow(self.column_names)
            CSV_writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        duplicate_indexes = []
        indexes = []
        rows = []
        for column in range(len(key_column_names)):
            indexes.append(self.column_names.index(key_column_names[column]))
        for i in range(len(self.data)):
            j = []
            for index in indexes:
                j.append(self.data[i][index])
            if j in rows:
                duplicate_indexes.append(i)
            else:
                rows.append(j)
        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        bad_rows = []
        for i in range(len(self.data)):
            if "NA" in self.data[i]:
                bad_rows.append(i)
        bad_rows.sort(reverse=True)
        for j in bad_rows:
            del (self.data[j])

    def replace_missing_values_with_column_average(self, col_name):
        for i in range(len(self.column_names)):
            if self.column_names[i] == col_name:
                column = i
        column_vals = []
        col_count = 0
        for i in range(len(self.data)):
            try:
                column_vals.append(float(self.data[i][column]))
                col_count += 1
            except:
                pass
        for i in range(len(self.data)):
            try:
                if self.data[i][column] == "NA":
                    self.data[i][column] = sum(column_vals)/col_count
            except:
                pass

    ########################################################################
    def compute_summary_statistics(self, col_names):
        table = []
        col_indexes = []
        for i in range(len(col_names)):
            for j in range(len(self.column_names)):
                if col_names[i] == self.column_names[j]:
                    col_indexes.append(j)

        if not self.data:
            table = []
            for i in range(len(col_indexes)):
                stats = []
                stats.append(self.column_names[col_indexes[i]])
                table.append(stats)
            new_table = MyPyTable(table, [])
            return new_table

        else:
            for i in range(len(col_indexes)):
                stats = []
                stats.append(self.column_names[col_indexes[i]])
                min = self.data[0][col_indexes[i]]
                for j in range(len(self.data)):
                    if self.data[j][col_indexes[i]] < min:
                        min = self.data[j][col_indexes[i]]
                stats.append(min)
                max = self.data[0][col_indexes[i]]
                for j in range(len(self.data)):
                    if self.data[j][col_indexes[i]] > max:
                        max = self.data[j][col_indexes[i]]
                stats.append(max)

                mid = (min + max) / 2
                stats.append(mid)

                # calculate average
                SUM = []
                for j in range(len(self.data)):
                    SUM.append(self.data[j][col_indexes[i]])
                avg = sum(SUM) / len(self.data)
                stats.append(avg)

                # calculate median

                sorted_list = []
                for row in range(len(self.data)):
                    sorted_list.append(self.data[row][col_indexes[i]])
                sorted_list.sort()
                if ((len(sorted_list) % 2) == 0):
                    upper_index = int(len(sorted_list) / 2)
                    median_upper = sorted_list[upper_index]
                    lower_index = int(len(sorted_list) / 2 - 1)
                    median_lower = sorted_list[lower_index]
                    median = (median_upper + median_lower) / 2
                    stats.append(median)
                else:

                    index = int((len(sorted_list) - 1) / 2)
                    median = sorted_list[index]
                    stats.append(median)
                table.append(stats)

            return MyPyTable([], table)  # TODO: fix this
########################################################################
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

    def perform_inner_join(self, other_table, key_column_names):
        col_names = []
        data = []
        col_copy = copy.deepcopy(other_table.column_names)
        for title in key_column_names:
            col_copy.remove(title)
        col_names = self.column_names + col_copy
        for row in self.data:
            keys = []
            for i in key_column_names:
                index = self.column_names.index(i)
                val = row[index]
                keys.append(val)
            for row2 in other_table.data:
                keys2 = []
                for j in key_column_names:
                    index = other_table.column_names.index(j)
                    val = row2[index]
                    keys2.append(val)
                if keys == keys2:
                    copy2 = copy.deepcopy(row2)
                    for val in keys2:
                        copy2.remove(val)
                    row_copy = row + copy2
                    data.append(row_copy)
        inner_join_table = MyPyTable(col_names, data)
        return inner_join_table

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
        return MyPyTable()  # TODO: fix this
