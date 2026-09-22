def process_one(job):
    """
    Runs in a worker process. Takes one (font_path, label, idx, text) job and
    returns a small dict of results instead of mutating shared state.
    """
    font_path, label, idx, text = job   # font path now travels with the job

    font = ImageFont.truetype(font_path, 20)
    lines = wrap_lines(text, font, max_width=5000)

    uppers_a, lowers_a = [], []
    for line in lines:
        u, l = analyze_text(line)
        uppers_a.extend(u)
        lowers_a.extend(l)

    img = lines_to_img(lines, font)
    res = compute_scores_att_3(np.array(img), call_idx=idx, debug=False)
    if res is None:
        return {"label": label, "failed": True}

    uppers_g_all, lowers_g_all = res

    return {
        "label": label,
        "failed": False,
        "uppers_a": uppers_a,
        "lowers_a": lowers_a,
        "uppers_g": uppers_g_all,
        "lowers_g": lowers_g_all,
    }