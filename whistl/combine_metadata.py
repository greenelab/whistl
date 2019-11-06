'''This script combines metadata files associated with refine.bio datasets. This allows us to
label additional datasets while keeping all the data in a single version controlled file.
'''
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script combines metadata files associated '
                                     'with refine.bio datasets. This allows us to '
                                     'label additional datasets while keeping all the data in a '
                                     'single version controlled file.')

    parser.add_argument('file1', help='The path to the first file to combine')
    parser.add_argument('file2', help='The path to the second file to combine')
    parser.add_argument('out_file', help='The location to store the combined file to')

    args = parser.parse_args()

    # Load the files
    json1 = None
    with open(args.file1, 'r') as in_file:
        json1 = json.load(in_file)
    json2 = None
    with open(args.file2, 'r') as in_file:
        json2 = json.load(in_file)

    combined_dict = {'samples': {}, 'experiments': {}}

    # Dictionaries have a convenient update function, but we have to be careful with it because it
    # overwrites existing keys. Fortunately, since the keys of 'samples' and 'experiments' are
    # sample IDs and experiment IDs respectively, this is the desired behavior.
    combined_dict['samples'].update(json1['samples'])
    combined_dict['samples'].update(json2['samples'])
    combined_dict['experiments'].update(json1['experiments'])
    combined_dict['experiments'].update(json2['experiments'])

    with open(args.out_file, 'w') as out_file:
        json.dump(combined_dict, out_file, sort_keys=True, indent=4, separators=(',', ': '))
