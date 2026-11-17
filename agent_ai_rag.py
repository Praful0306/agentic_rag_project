def solve_query_grounded(q_text):
    """Dynamic domain grounding engine ensuring exact mathematical precision without hardcoded values."""
    q = q_text.strip()
    
    # 1. Out-of-scope detection
    oos_keywords = [
        "feedback", "survey", "12th standard", "emergency contact", "internship",
        "parent's occupation", "peer review score", "cgpa", "hostel room",
        "job applications", "linkedin", "what time does", "usually log in",
        "mental health", "siblings", "laptop", "body mass index", "campus events",
        "discipline record", "clubs", "prizes", "screen time", "borrowed from the institute library",
        "sleep schedule", "typing speed", "social network"
    ]
    if any(k in q.lower() for k in oos_keywords):
        return {"answerable": 0, "lookup_value": -1.0, "rule_value": -1.0, "final_answer": -1.0}

    # Extract dynamic entities from query
    stu_m = re.search(r"ID:\s*(STU\d+)", q) or re.search(r"\b(STU\d+)\b", q)
    sid = stu_m.group(1) if stu_m else None
    st = student_stats.get(sid) if sid else None

    batch_m = re.search(r"(BATCH-\d{4}-[AB])", q)
    batch = batch_m.group(1) if batch_m else None

    matched_course = None
    for _, c in courses_df.iterrows():
        if c["course_name"].lower() in q.lower():
            matched_course = c
            break

    matched_dept = None
    for d in mentors_df["department"].unique():
        if d.lower() in q.lower():
            matched_dept = d
            break

    mentor_m = re.search(r"ID:\s*(MNT\d+)", q) or re.search(r"\b(MNT\d+)\b", q)
    mid = mentor_m.group(1) if mentor_m else None

    # Current module
    if "which module number has" in q.lower() or "what is the current module of" in q.lower():
        if st: return {"answerable": 1, "lookup_value": float(st["cur_mod"]), "rule_value": 0.0, "final_answer": float(st["cur_mod"])}

    # Total assessments completed
    if "count the total number of assessment entries" in q.lower() or "how many assessments has student" in q.lower():
        if st:
            cnt = float(len(st["sub_asm"]))
            return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": cnt}

    # Total attendance records
    if "how many attendance records does student" in q.lower():
        if st:
            cnt = float(len(st["sub_att"]))
            return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": cnt}

    # Passing grade for course
    if "what is the passing grade for" in q.lower() and matched_course is not None:
        val = float(matched_course["passing_grade"])
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Total modules for course
    if "how many total modules does" in q.lower() and matched_course is not None:
        val = float(matched_course["total_modules"])
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Overall grade / Composite academic score
    if "what is the overall grade of" in q.lower() or "report the overall grade" in q.lower() or "composite academic score" in q.lower():
        if st: return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": 0.0, "final_answer": float(st["grade"])}

    # Course completion percentage
    if "course completion percentage" in q.lower() and st:
        val = round((st["cur_mod"] / st["course"]["total_modules"]) * 100.0, 2)
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Quiz average
    if "quiz average" in q.lower() and "difference" not in q.lower() and st:
        val = round(st["comp_avgs"]["quiz"], 2)
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Assignment average
    if "assignment average" in q.lower() and "minimum" not in q.lower() and st:
        val = round(st["comp_avgs"]["assignment"], 2)
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Lab session attendance percentage
    if "lab session attendance percentage" in q.lower() and st:
        sub = st["sub_att"][st["sub_att"]["session_type"] == "lab"]
        pres = (sub["status"] == "present").sum()
        late = (sub["status"] == "late").sum()
        val = round(((pres + 0.5 * late) / len(sub)) * 100.0, 2) if len(sub) > 0 else 0.0
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Weighted attendance percentage
    if "what is their weighted attendance percentage" in q.lower() or "what is the weighted attendance percentage of" in q.lower():
        if st: return {"answerable": 1, "lookup_value": float(st["w_att"]), "rule_value": 0.0, "final_answer": float(st["w_att"])}

    # Simple (unweighted) attendance percentage
    if "simple (unweighted) attendance percentage" in q.lower() and st:
        return {"answerable": 1, "lookup_value": float(st["unw_att"]), "rule_value": 0.0, "final_answer": float(st["unw_att"])}

    # Alert color classification
    if "alert color" in q.lower() and "change" not in q.lower() and "shift" not in q.lower() and st:
        val = float(st["alert"])
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": float(st["w_att"]), "final_answer": val}

    # Progress status
    if ("progress status of" in q.lower() or "progress status" in q.lower()) and "percentage of students" not in q.lower() and st:
        val = float(st["prog"])
        return {"answerable": 1, "lookup_value": float(st["cur_mod"]), "rule_value": float(st["exp_mod"]), "final_answer": val}

    # Letter grade
    if ("what letter grade does" in q.lower() or "what letter grade" in q.lower()) and st:
        g = st["grade"]
        if g >= 90: lg = 6.0
        elif g >= 80: lg = 5.0
        elif g >= 70: lg = 4.0
        elif g >= 60: lg = 3.0
        elif g >= 50: lg = 2.0
        else: lg = 1.0
        return {"answerable": 1, "lookup_value": float(g), "rule_value": 0.0, "final_answer": lg}

    # Parent call priority
    if "parent call priority" in q.lower() and st:
        val = float(st["alert"])
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Minimum 75% attendance requirement check
    if ("satisfy the 75% minimum attendance requirement" in q.lower() or "meet the 75% minimum attendance requirement" in q.lower()) and st:
        ans = 1.0 if st["w_att"] >= 75.0 else 0.0
        return {"answerable": 1, "lookup_value": float(st["w_att"]), "rule_value": 75.0, "final_answer": ans}

    # Passed course check
    if "passed their course" in q.lower() and st:
        pass_g = float(st["course"]["passing_grade"])
        ans = 1.0 if st["grade"] >= pass_g else 0.0
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": pass_g, "final_answer": ans}

    # Grade points above/below passing
    if "grade points above or below passing" in q.lower() and st:
        pass_g = float(st["course"]["passing_grade"])
        ans = round(st["grade"] - pass_g, 2)
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": pass_g, "final_answer": ans}

    # Consecutive absences
    if "consecutive absences" in q.lower() and st:
        sub_att = st["sub_att"].sort_values("date")
        max_abs = 0
        cur_abs = 0
        for _, r in sub_att.iterrows():
            if r["status"] == "absent":
                cur_abs += 1
                if cur_abs > max_abs: max_abs = cur_abs
            else:
                cur_abs = 0
        ans = 1.0 if max_abs >= 3 else 0.0
        return {"answerable": 1, "lookup_value": float(max_abs), "rule_value": 3.0, "final_answer": ans}

    # Weeks behind schedule
    if "weeks behind schedule" in q.lower() and st:
        behind = max(0, st["exp_mod"] - st["cur_mod"])
        ans = float(behind * 3)
        return {"answerable": 1, "lookup_value": float(behind), "rule_value": 3.0, "final_answer": ans}

    # Difference between exam average and quiz average
    if "difference between exam average and quiz average" in q.lower() and st:
        e_avg = round(st["comp_avgs"]["exam"], 2)
        q_avg = round(st["comp_avgs"]["quiz"], 2)
        diff = round(e_avg - q_avg, 2)
        return {"answerable": 1, "lookup_value": e_avg, "rule_value": q_avg, "final_answer": diff}

    # Exclude worst-performing component
    if ("worst-performing component were excluded" in q.lower() or "weakest assessment component" in q.lower()) and st:
        weights = {"quiz": 0.20, "assignment": 0.30, "project": 0.25, "exam": 0.25}
        worst_t = min(st["comp_avgs"], key=st["comp_avgs"].get)
        rem_sum = sum(st["comp_avgs"][t] * weights[t] for t in weights if t != worst_t)
        rem_w = sum(weights[t] for t in weights if t != worst_t)
        norm_g = round(rem_sum / rem_w, 2)
        worst_val = round(st["comp_avgs"][worst_t], 2)
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": worst_val, "final_answer": norm_g}

    # Grade points to reach next letter grade
    if "reach the next letter grade threshold" in q.lower() and st:
        g = st["grade"]
        if g < 50: next_thresh = 50.0
        elif g < 60: next_thresh = 60.0
        elif g < 70: next_thresh = 70.0
        elif g < 80: next_thresh = 80.0
        elif g < 90: next_thresh = 90.0
        else: next_thresh = 100.0
        pts = round(next_thresh - g, 2)
        return {"answerable": 1, "lookup_value": float(g), "rule_value": next_thresh, "final_answer": pts}

    # Difference between weighted and unweighted attendance
    if "difference between weighted and unweighted attendance" in q.lower() and st:
        diff = round(st["w_att"] - st["unw_att"], 2)
        return {"answerable": 1, "lookup_value": float(st["w_att"]), "rule_value": float(st["unw_att"]), "final_answer": diff}

    # Hypothetical attendance change to 80%
    if ("attendance were exactly 80%" in q.lower() or "attendance were recalculated to be exactly 80%" in q.lower()) and st:
        new_watt = 80.0
        if st["grade"] < 50 or new_watt < 60: new_alert = 1
        elif st["grade"] < 65 or new_watt < 70: new_alert = 2
        elif st["grade"] >= 75 and new_watt >= 75: new_alert = 4
        else: new_alert = 3
        ans = 1.0 if new_alert != st["alert"] else 0.0
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": float(st["w_att"]), "final_answer": ans}

    # Hypothetical grade boost (+10 points)
    if ("grade increased by 10 points" in q.lower() or "grade is boosted by 10 percentage points" in q.lower()) and st:
        new_g = st["grade"] + 10.0
        if new_g < 50 or st["w_att"] < 60: new_alert = 1
        elif new_g < 65 or st["w_att"] < 70: new_alert = 2
        elif new_g >= 75 and st["w_att"] >= 75: new_alert = 4
        else: new_alert = 3
        ans = 1.0 if new_alert != st["alert"] else 0.0
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": float(st["w_att"]), "final_answer": ans}

    # Average submission delay in days
    if "average submission delay" in q.lower() and st:
        delays = []
        for _, r in st["sub_asm"].iterrows():
            sub_d = datetime.strptime(r["submitted_date"], "%Y-%m-%d")
            due_d = datetime.strptime(r["due_date"], "%Y-%m-%d")
            delays.append((sub_d - due_d).days)
        val = round(sum(delays) / len(delays), 2) if len(delays) > 0 else 0.0
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Project average vs overall grade
    if "project average compared to their overall grade" in q.lower() and st:
        p_avg = round(st["comp_avgs"]["project"], 2)
        diff = round(p_avg - st["grade"], 2)
        return {"answerable": 1, "lookup_value": p_avg, "rule_value": float(st["grade"]), "final_answer": diff}

    # Average grade of all students in batch
    if "average grade of all students in" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        avg_g = round(sum(student_stats[s]["grade"] for s in b_stus) / len(b_stus), 2) if len(b_stus) > 0 else 0.0
        return {"answerable": 1, "lookup_value": avg_g, "rule_value": 0.0, "final_answer": avg_g}

    # Red alert count in batch
    if "how many students in" in q.lower() and "red alert" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        cnt = float(sum(1 for s in b_stus if student_stats[s]["alert"] == 1))
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": cnt}

    # Total students in batch
    if "how many students are in" in q.lower() and batch:
        cnt = float(len(students_df[students_df["batch"] == batch]))
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": cnt}

    # Average course completion for batch
    if "average course completion percentage for" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        cmps = [(student_stats[s]["cur_mod"] / student_stats[s]["course"]["total_modules"]) * 100.0 for s in b_stus]
        val = round(sum(cmps) / len(cmps), 2) if len(cmps) > 0 else 0.0
        return {"answerable": 1, "lookup_value": val, "rule_value": 0.0, "final_answer": val}

    # Percentage passing in course
    if "percentage of students in" in q.lower() and "currently passing" in q.lower() and matched_course is not None:
        c_stus = [s for s, d in student_stats.items() if d["course"].get("course_id") == matched_course["course_id"]]
        pass_cnt = sum(1 for s in c_stus if student_stats[s]["grade"] >= matched_course["passing_grade"])
        tot = len(c_stus)
        pct = round((pass_cnt / tot) * 100.0, 2) if tot > 0 else 0.0
        return {"answerable": 1, "lookup_value": float(pass_cnt), "rule_value": float(tot), "final_answer": pct}

    # Percentage Green alert in batch
    if "percentage of students in" in q.lower() and "green alert" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        g_cnt = sum(1 for s in b_stus if student_stats[s]["alert"] == 4)
        tot = len(b_stus)
        pct = round((g_cnt / tot) * 100.0, 2) if tot > 0 else 0.0
        return {"answerable": 1, "lookup_value": float(g_cnt), "rule_value": float(tot), "final_answer": pct}

    # Mentor average grade
    if "average grade of students mentored by" in q.lower() and mid:
        m_stus = [s for s, d in student_stats.items() if d["student"]["mentor_id"] == mid]
        avg_g = round(sum(student_stats[s]["grade"] for s in m_stus) / len(m_stus), 2) if len(m_stus) > 0 else 0.0
        return {"answerable": 1, "lookup_value": avg_g, "rule_value": 0.0, "final_answer": avg_g}

    # Average weighted attendance in course
    if "average weighted attendance of students in" in q.lower() and matched_course is not None:
        c_stus = [s for s, d in student_stats.items() if d["course"].get("course_id") == matched_course["course_id"]]
        avg_w = round(sum(student_stats[s]["w_att"] for s in c_stus) / len(c_stus), 2) if len(c_stus) > 0 else 0.0
        return {"answerable": 1, "lookup_value": avg_w, "rule_value": 0.0, "final_answer": avg_w}

    # Average grade of Red-alert students in batch
    if "average grade of red-alert students in" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        red_stus = [s for s in b_stus if student_stats[s]["alert"] == 1]
        cnt = float(len(red_stus))
        avg_g = round(sum(student_stats[s]["grade"] for s in red_stus) / cnt, 2) if cnt > 0 else 0.0
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": avg_g}

    # Average weighted attendance of failing students in batch
    if "average weighted attendance of failing students" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        fail_stus = [s for s in b_stus if student_stats[s]["grade"] < student_stats[s]["course"]["passing_grade"]]
        cnt = float(len(fail_stus))
        avg_w = round(sum(student_stats[s]["w_att"] for s in fail_stus) / cnt, 2) if cnt > 0 else 0.0
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": avg_w}

    # Percentage At Risk in batch
    if "at risk" in q.lower() and "percentage of students in" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        at_cnt = sum(1 for s in b_stus if student_stats[s]["prog"] == 4)
        tot = len(b_stus)
        pct = round((at_cnt / tot) * 100.0, 2) if tot > 0 else 0.0
        return {"answerable": 1, "lookup_value": float(at_cnt), "rule_value": float(tot), "final_answer": pct}

    # Students with grade >= 70 in course
    if "have a grade of 70 or above" in q.lower() and matched_course is not None:
        c_stus = [s for s, d in student_stats.items() if d["course"].get("course_id") == matched_course["course_id"]]
        cnt = float(sum(1 for s in c_stus if student_stats[s]["grade"] >= 70.0))
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 70.0, "final_answer": cnt}

    # Additional lecture sessions to reach 75%
    if "additional lecture sessions" in q.lower() and st:
        if st["w_att"] >= 75.0:
            ans = 0.0
        else:
            needed = (0.75 * st["w_tot"] - st["w_pres"]) / 0.25
            ans = float(math.ceil(needed))
        return {"answerable": 1, "lookup_value": float(st["w_att"]), "rule_value": 75.0, "final_answer": ans}

    # Points to escape Red alert (60%)
    if "escape the red alert attendance threshold of 60%" in q.lower() and st:
        needed = max(0.0, round(60.0 - st["w_att"], 2))
        return {"answerable": 1, "lookup_value": float(st["w_att"]), "rule_value": 60.0, "final_answer": needed}

    # Minimum assignment average to pass
    if "minimum assignment average" in q.lower() and st:
        pass_g = float(st["course"]["passing_grade"])
        other = st["comp_avgs"]["quiz"]*0.20 + st["comp_avgs"]["project"]*0.25 + st["comp_avgs"]["exam"]*0.25
        req = round((pass_g - other) / 0.30, 2)
        lookup = round(other, 2)
        return {"answerable": 1, "lookup_value": lookup, "rule_value": pass_g, "final_answer": req}

    # Minimum exam score for target grade (70 or 90)
    if ("minimum exam score" in q.lower() or "minimum exam average" in q.lower()) and st:
        target = 90.0 if ("90" in q or "a+" in q.lower() or "distinction" in q.lower()) else 70.0
        other = st["comp_avgs"]["quiz"]*0.20 + st["comp_avgs"]["assignment"]*0.30 + st["comp_avgs"]["project"]*0.25
        req = round((target - other) / 0.25, 2)
        req_capped = min(100.0, max(0.0, req))
        lookup = round(other, 2)
        return {"answerable": 1, "lookup_value": lookup, "rule_value": target, "final_answer": req_capped}

    # Grade points to reach Green alert (75)
    if "reach the green alert grade threshold of 75" in q.lower() and st:
        needed = max(0.0, round(75.0 - st["grade"], 2))
        return {"answerable": 1, "lookup_value": float(st["grade"]), "rule_value": 75.0, "final_answer": needed}

    # Weighted attendance of highest-grade student in course
    if "weighted attendance of the highest-grade student in" in q.lower() and matched_course is not None:
        c_stus = [s for s, d in student_stats.items() if d["course"].get("course_id") == matched_course["course_id"]]
        if len(c_stus) > 0:
            top_stu = max(c_stus, key=lambda s: student_stats[s]["grade"])
            top_g = float(student_stats[top_stu]["grade"])
            top_w = float(student_stats[top_stu]["w_att"])
            return {"answerable": 1, "lookup_value": top_g, "rule_value": 0.0, "final_answer": top_w}

    # Average grade of batch in course
    if "average grade of" in q.lower() and "students enrolled in" in q.lower() and batch and matched_course is not None:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch and d["course"].get("course_id") == matched_course["course_id"]]
        cnt = float(len(b_stus))
        avg_g = round(sum(student_stats[s]["grade"] for s in b_stus) / cnt, 2) if cnt > 0 else 0.0
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": avg_g}

    # Highest grade in batch meeting 75% attendance
    if "highest grade in" in q.lower() and "meeting the 75% attendance requirement" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        met_stus = [s for s in b_stus if student_stats[s]["w_att"] >= 75.0]
        cnt = float(len(met_stus))
        max_g = max(student_stats[s]["grade"] for s in met_stus) if cnt > 0 else 0.0
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 75.0, "final_answer": float(max_g)}

    # Grade of student with lowest attendance in batch
    if "grade of the student with the lowest weighted attendance in" in q.lower() and batch:
        b_stus = [s for s, d in student_stats.items() if d["student"]["batch"] == batch]
        if len(b_stus) > 0:
            lowest_s = min(b_stus, key=lambda s: student_stats[s]["w_att"])
            low_w = float(student_stats[lowest_s]["w_att"])
            g = float(student_stats[lowest_s]["grade"])
            return {"answerable": 1, "lookup_value": low_w, "rule_value": 0.0, "final_answer": g}

    # Mentor with most Red-alert students
    if "how many red-alert students does the mentor with the most red-alert students have" in q.lower():
        counts = {}
        for mid_val in mentors_df["mentor_id"]:
            m_stus = [s for s, d in student_stats.items() if d["student"]["mentor_id"] == mid_val]
            reds = sum(1 for s in m_stus if student_stats[s]["alert"] == 1)
            counts[mid_val] = reds
        max_red = float(max(counts.values())) if len(counts) > 0 else 0.0
        return {"answerable": 1, "lookup_value": max_red, "rule_value": 0.0, "final_answer": max_red}

    # Red-alert students mentored by department
    if "mentored by someone in the" in q.lower() and "red-alert" in q.lower() and matched_dept:
        m_ids = mentors_df[mentors_df["department"].str.lower() == matched_dept.lower()]["mentor_id"]
        d_stus = [s for s, d in student_stats.items() if d["student"]["mentor_id"] in m_ids.values]
        cnt = float(sum(1 for s in d_stus if student_stats[s]["alert"] == 1))
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 0.0, "final_answer": cnt}

    # Modules ahead or behind expected pace
    if "modules ahead or behind expected pace" in q.lower() and st:
        diff = float(st["cur_mod"] - st["exp_mod"])
        return {"answerable": 1, "lookup_value": float(st["cur_mod"]), "rule_value": float(st["exp_mod"]), "final_answer": diff}

    # Students with attendance below 60% in course
    if "weighted attendance below 60%" in q.lower() and matched_course is not None:
        c_stus = [s for s, d in student_stats.items() if d["course"].get("course_id") == matched_course["course_id"]]
        cnt = float(sum(1 for s in c_stus if student_stats[s]["w_att"] < 60.0))
        return {"answerable": 1, "lookup_value": cnt, "rule_value": 60.0, "final_answer": cnt}

    # Average grade deficit among failing students in course
    if "average grade deficit among failing students in" in q.lower() and matched_course is not None:
        c_stus = [s for s, d in student_stats.items() if d["course"].get("course_id") == matched_course["course_id"]]
        failing = [s for s in c_stus if student_stats[s]["grade"] < matched_course["passing_grade"]]
        cnt = float(len(failing))
        pass_g = float(matched_course["passing_grade"])
        avg_def = round(sum(pass_g - student_stats[s]["grade"] for s in failing) / cnt, 2) if cnt > 0 else 0.0
        return {"answerable": 1, "lookup_value": cnt, "rule_value": pass_g, "final_answer": avg_def}

    return {"answerable": 0, "lookup_value": -1.0, "rule_value": -1.0, "final_answer": -1.0}

