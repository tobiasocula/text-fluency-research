

def allsublists(arr):

    if len(arr) == 1:
        return [arr]

    right = allsublists(arr[1:])
    left = allsublists(arr[:-1])
    second_val = arr[1]
    secondlast_val = arr[:-2]
    res = []
    for r in right:
        res.append(r)
        if r[0] == second_val:
            res.append([second_val] + r)

    for l in left:
        res.append(l)
        if l[-1] == secondlast_val:
            res.append(l + [secondlast_val])

    return res

