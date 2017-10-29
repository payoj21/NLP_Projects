def pprint_double_dict(d, column_width = 5, max_chars = 100):
    rows = sorted(d.keys())
    columns = sorted(d[rows[0]].keys())
    max_allowed_columns = (max_chars // column_width) - 1
    total_columns = len(columns)
    cuts = total_columns // max_allowed_columns

    if cuts == 0:
        ranges = [(0, total_columns)]
    else:
        ranges = [(i*max_allowed_columns, i*max_allowed_columns+max_allowed_columns) for i in range(cuts)]
        if total_columns >= ranges[-1][1]:
            ranges.append((ranges[-1][1], total_columns))

    for each_range in ranges:
        print("\n")
        print("{:<{width}}".format("", width=column_width), end="")
        for column in columns[each_range[0] : each_range[1]]:
            print("{:<{width}}".format(str(column), width=column_width), end="")
        print("")

        for row in rows:
            print("{:<{width}}".format(str(row), width=column_width), end="")
            for column in columns[each_range[0] : each_range[1]]:
                print("{:<{width}}".format(str(d[row][column]), width=column_width), end="")
            print("")


def transition_matrix(tag_dictionary):
    pos_tags = list(tag_dictionary.keys())

    for row in pos_tags:
        total_probability = 0
        for column in pos_tags:
            total_probability += tag_dictionary[column]['Probability_after_smoothing'][row]
        print(row, " ", total_probability)

def emission_matrix(word_dictionary,tag_dictionary):
    word = list(word_dictionary.keys())
    tag = list(tag_dictionary.keys())
    for row in tag:
        total_probability = 0
        for column in word:
            if row in word_dictionary[column]['Tags']:
                total_probability += word_dictionary[column]['Tags'][row]['Probability']
        print(row, " ", total_probability)