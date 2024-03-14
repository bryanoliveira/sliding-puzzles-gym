def count_inversions(arr):
    # Count inversions in the array (exclude the blank tile)
    # Time complexity: O(n^2) = O(w*h)
    inv_count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count

def is_solvable(inversions, blank_row, width, height):
    # Check if the puzzle configuration is solvable
    if width % 2 != 0:  # Odd width
        return inversions % 2 == 0
    else:  # Even width
        # For even width, the puzzle is solvable if the blank is on an even row counting from the bottom
        # and the number of inversions is odd, or vice versa.
        return (inversions + height - blank_row) % 2 != 0

def inverse_action(action):
    return {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
    }.get(action, 4)