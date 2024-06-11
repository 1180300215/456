# import os, sys
# from pprint import PrettyPrinter
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from deployment_stage.collect_data import collect_data
# import pickle
#
#
# def main():
#     all_data = collect_data()
#     with open("all_online_data.pkl",'wb') as fo:
#         pickle.dump(all_data, fo)
#
#
# if __name__ == '__main__':
#     main()
#
import pickle
with open('all_online_data.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data))