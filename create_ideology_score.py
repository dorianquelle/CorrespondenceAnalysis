
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from scipy.sparse import csc_matrix
from collections import Counter
import argparse
import datetime
import time

starting_time = datetime.datetime.now()
# Create the parser
parser = argparse.ArgumentParser()

# Add the filename argument with default value and a help message
parser.add_argument("--filename", default="tuples.pickle", help="The name of the file to process (pickle dict)")

# Add the minimum number of occurrences for the user and forum arguments, with default values and help messages
parser.add_argument("--min_user", default=100, type=int, help="Minimum number of occurrences user")
parser.add_argument("--min_forum", default=100, type=int, help="Minimum number of occurrences forum")

# Add the verbose flag argument with default value and a help message
parser.add_argument("--verbose", default=True, type=bool, help="Verbose")

# Add the output file argument with default value and a help message
parser.add_argument("--outfile", default="../data/results.pickle", help="Outfile")

# Parse the command-line arguments and assign them to local variables
args = parser.parse_args()

filename, min_user, min_forum, verbose, outfile = args.filename, args.min_user, args.min_forum, args.verbose, args.outfile

# Define a helper function to flatten a list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# Open the file with the given filename in binary mode
with open(filename, "rb") as f:
    # Load the pickle dictionary from the file
    tups = pickle.load(f)

# Filter the dictionary to only include keys that have a value with at least `min_forum` elements
tups = {k: v for (k, v) in tups.items() if len(v) >= min_forum}

# Count the number of occurrences of each user in the values of the dictionary
user_counts = Counter(flatten(list(tups.values())))

# Filter the user counts to only include users that have at least `min_user` occurrences
user_counts = {x: count for x, count in user_counts.items() if count >= min_user}

# Create a set of users to keep, based on the filtered user counts
user_to_keep = set(list(user_counts.keys()))

# This removes Guests. 
user_to_keep = user_to_keep - set([None]) # Update this!

# Iterate over the keys in the dictionary
for tup in tqdm(tups.keys()):
    # Filter the value for the current key to only include users that are in the user_to_keep set
    tups[tup] = [x for x in tups[tup] if x in user_to_keep]

# Create a list of unique forums from the keys of the dictionary
forums = list(set(list(tups.keys())))

# Create a list of unique users from the values of the dictionary
user = list(set(flatten(list(tups.values()))))

# Print the number of users and forums after filtering
if verbose: print(f"After Filtering we have {len(user):,} Users and {len(forums):,} Forums")

# Create a dictionary mapping forum names to unique ids
forums_id = {x: i for i, x in enumerate(forums)}

# Create a dictionary mapping user names to unique ids
user_id = {x: i for i, x in enumerate(user)}

# Print a message indicating that the adjacency matrix is being created
if verbose: print("Create Adjacency Matrix")

# Create a zero matrix with the number of rows equal to the number of users and the number of columns equal to the number of forums
mat = np.zeros(shape=(len(user_id.keys()), len(forums_id.keys())))

# Iterate over the keys in the dictionary
for tup in tqdm(list(tups.keys())):
    # Get the list of users for the current key (forum)
    users_in_forum = tups[tup]
    
    # Get the unique id for the current forum
    forum_id = forums_id[tup]
    
    # Iterate over the users in the current forum
    for user_ in users_in_forum:
        # Increment the element in the adjacency matrix at the row corresponding to the current user and the column corresponding to the current forum
        mat[user_id[user_], forum_id] += 1

# Convert the adjacency matrix to a compressed sparse column (CSC) matrix
adj_matrix = csc_matrix(mat)

# Print a message indicating that the matrix is being normalized
if verbose: print("Starting to Normalize Matrix")

# Calculate the total sum of the adjacency matrix
TOTAL_SUM = adj_matrix.sum()

# Divide the adjacency matrix by its total sum to create a probability matrix
P = adj_matrix / TOTAL_SUM

# Delete the adjacency matrix and the total sum to free up memory
del adj_matrix
del TOTAL_SUM

# Calcul# Calculate the row mass of the probability matrix
r = P.sum(axis=1)

# Create a diagonal matrix from the flattened row mass vector
Dr = np.diag(np.array(flatten(r.tolist())))

# Calculate the column mass of the probability matrix
c = P.sum(axis=0)

# Create a diagonal matrix from the flattened column mass vector
Dc = np.diag(np.array(flatten(c.tolist())).T)

# Print a message indicating that the standardized matrix is being calculated
if verbose: print("Calculating Standardized")

# Calculate the standardized matrix by subtracting the product of the row and column mass matrices from the probability matrix
standardized = (P - (r.dot(c)))

# Print a message indicating that the S matrix is being calculated
if verbose: print("Calculating S")

# Calculate the square root of the Dr matrix
sDr = np.sqrt(Dr)

# Calculate the square root of the Dc matrix
sDc = np.sqrt(Dc)

# Calculate the S matrix by multiplying the square root of the Dr matrix by the standardized matrix and then by the square root of the Dc matrix
S = sDr.dot(standardized).dot(sDc)

# Print a message indicating that singular value decomposition (SVD) is starting
if verbose: print("Starting SVD")

# Use numpy's SVD function to calculate the matrices u, d, and v from the S matrix
u, d, v = np.linalg.svd(S)

# Print a message indicating that the projection is starting
if verbose: print("Starting Projection")

phi = sDr.dot(u)
gamma = sDc.dot(v)


# Create Loadings for both
forum_ideo = pd.DataFrame(np.concatenate([gamma[0,:],np.array(forums)[np.newaxis]], axis = 0)).transpose()
user_ideo = pd.DataFrame(np.concatenate([phi[:,0].T,np.array(user)[np.newaxis]])).transpose()
forum_ideo.columns = ["ideo","forum"]
user_ideo.columns = ["ideo","user"]

out = {"phi": phi,  "gamma": gamma,  "u": u ,  "d": d,
       "v": v, "forums": forums_id, "user": user_id,
      "forum_ideo": forum_ideo, "user_ideo": user_ideo}

# Open the file for writing
with open(outfile, "wb") as f:
    # Dump content. 
    pickle.dump(out,f)
  

end_time = datetime.datetime.now()
delta = end_time - starting_time
diff_time = delta.total_seconds()

if verbose: 
    print()
    print("#"*90)
    print(f"Done! Data saved in {outfile}")
    print(f"Total Execution time: {diff_time} (Seconds)")
    print("#"*90)

if verbose:
    print("Report:")
    print("#"*90)
    print("The top 20 Forums by Ideology are:")
    for i,t in enumerate(forum_ideo.sort_values("ideo").head(20).forum.values):
        print(f"Forum {i}: {t}", end = "\n")
    print("\n")

    print()
    print("The lowest 20 Forums by Ideology are:")
    for i,t in enumerate(reversed(forum_ideo.sort_values("ideo").tail(20).forum.values)):
        print(f"Forum {i}: {t}", end = "\n")
    print("\n\n\n\n")

    print("The top 20 Users by Ideology are:")
    for i,t in enumerate(user_ideo.sort_values("ideo").head(20).user.values):
        print(f"User {i}: {t}", end = "\n")
    print("\n")

    print("The lowest 20 Forums by Ideology are:")
    for i,t in enumerate(reversed(user_ideo.sort_values("ideo").tail(20).user.values)):
        print(f"User {i}: {t}", end = "\n")
    print("#"*90)

