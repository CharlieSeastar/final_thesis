import json
import os
import numpy as np

subject_root = "courseMetadata"
student_link_prefix = "https://www.classcentral.com"

course_map = {}
student_map = {}
student_groups_gender = {"male": [], "female": []}

student_index_url = {}
student_url_index = {}

course_index_url = {}
course_url_index = {}


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        return json.load(fin)


def get_files(file_dir):
    json_files = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(root + '/' + file)
    return json_files


def get_course_subject():
    subjects = []
    for _, dirs, _ in os.walk(subject_root):
        for dir in dirs:
            subjects.append(dir)
    return subjects


def total_err(r, p, q):
    err_sum = 0.0
    count = 0
    for course_url in course_map:
        row_index = course_url_index[course_url]
        for student_url, rating in course_map[course_url]["ratings"]:
            count += 1
            col_index = student_url_index[student_url]

            user_bias = student_map[student_url]["rating_bias"]
            item_bias = course_map[course_url]["rating_bias"]

            predict = float(np.dot(q[row_index], p[col_index]))
            err = float(r[row_index][col_index]) - predict - user_bias - item_bias
            err_sum += err ** 2
    return round(err_sum / count, 3)


def e_g_item_predict(p, q, course_url):
    adv_group = 0
    disadv_group = 0
    adv_count = 0
    disadv_count = 0
    row_index = course_url_index[course_url]
    item_bias = course_map[course_url]["rating_bias"]
    for student_url, rating in course_map[course_url]["ratings"]:
        col_index = student_url_index[student_url]
        gender = student_map[student_url]
        predict = float(np.dot(q[row_index], p[col_index])) + student_map[student_url]["rating_bias"] + item_bias
        if gender == "male":
            adv_count += 1
            adv_group += predict
        else:
            disadv_count += 1
            disadv_group += predict
    if adv_count == 0:
        adv_group = 0
    else:
        adv_group /= adv_count
    if disadv_count == 0:
        disadv_group = 0
    else:
        disadv_group /= disadv_count
    return disadv_group, adv_group


def e_g_item_real(r, course_url):
    adv_group = 0
    disadv_group = 0
    adv_count = 0
    disadv_count = 0
    row_index = course_url_index[course_url]
    for student_url, rating in course_map[course_url]["ratings"]:
        col_index = student_url_index[student_url]
        gender = student_map[student_url]["gender"]
        value = r[row_index][col_index]
        if gender == "male":
            adv_count += 1
            adv_group += value
        else:
            disadv_count += 1
            disadv_group += value
    if adv_count == 0:
        adv_group = 0
    else:
        adv_group /= adv_count
    if disadv_count == 0:
        disadv_group = 0
    else:
        disadv_group /= disadv_count
    return disadv_group, adv_group


def fair_val_single(r, p, q, course_url):
    predict = e_g_item_predict(p, q, course_url)
    real = e_g_item_real(r, course_url)
    return abs(predict[0] - real[0] - predict[1] + real[1])


def fair_val_all(r, p, q):
    total = 0
    for course_url in course_map:
        total += fair_val_single(r, p, q, course_url)
    return round(total / len(course_map), 3)


def fair_abs_single(r, p, q, course_url):
    predict = e_g_item_predict(p, q, course_url)
    real = e_g_item_real(r, course_url)
    return abs(abs(predict[0] - real[0]) - abs(predict[1] - real[1]))


def fair_abs_all(r, p, q):
    total = 0
    for course_url in course_map:
        total += fair_abs_single(r, p, q, course_url)
    return round(total / len(course_map), 3)


def fair_under_single(r, p, q, course_url):
    predict = e_g_item_predict(p, q, course_url)
    real = e_g_item_real(r, course_url)
    return abs(max(0, real[0] - predict[0]) - max(0, real[1] - predict[1]))


def fair_under_all(r, p, q):
    total = 0
    for course_url in course_map:
        total += fair_under_single(r, p, q, course_url)
    return round(total / len(course_map), 3)


def fair_over_single(r, p, q, course_url):
    predict = e_g_item_predict(p, q, course_url)
    real = e_g_item_real(r, course_url)
    return abs(max(0, predict[0] - real[0]) - max(0, predict[1] - real[1]))


def fair_over_all(r, p, q):
    total = 0
    for course_url in course_map:
        total += fair_over_single(r, p, q, course_url)
    return round(total / len(course_map), 3)


def fair_par(r, p, q):
    adv_group = 0
    disadv_group = 0
    adv_count = 0
    disadv_count = 0
    for course_url in course_map:
        row_index = course_url_index[course_url]
        item_bias = course_map[course_url]["rating_bias"]
        for student_url, rating in course_map[course_url]["ratings"]:
            col_index = student_url_index[student_url]
            gender = student_map[student_url]["gender"]
            predict = float(np.dot(q[row_index], p[col_index])) + student_map[student_url]["rating_bias"] + item_bias
            if gender == "male":
                adv_count += 1
                adv_group += predict
            else:
                disadv_count += 1
                disadv_group += predict
    return round(abs(disadv_group / disadv_count - adv_group / adv_count), 3)


def huber_loss(e, d=1):
    if abs(e) <= d:
        return e ** 2 / 2
    else:
        return d * (abs(e) - d / 2)


if __name__ == '__main__':
    gender_info = read_json("gender_info/gender.json")

    student_count = 0
    course_count = 0
    total_review = 0
    total_rating = 0
    for file_path in get_files("courseReview"):
        review_info = read_json(file_path)
        course_url = review_info[0]["courseURL"]
        course_map[course_url] = {}
        course_map[course_url]["ratings"] = []
        course_count += 1
        course_rating_total = 0
        for review in review_info:
            student_link = review["studentLink"]
            rating = int(review["rating"])
            course_rating_total += rating
            if student_link == "":
                continue
            student_url = student_link_prefix + student_link
            if student_map.get(student_url) is None:
                student_count += 1
                tmp = gender_info.get(student_url)
                gender = 'male'
                if tmp is None or tmp[1] != 'male':
                    gender = 'female'
                student_map[student_url] = {}
                student_map[student_url]["ratings"] = [(course_url, rating)]
                student_map[student_url]["gender"] = gender
                student_groups_gender[gender].append(student_url)
            else:
                student_map[student_url]["ratings"].append((course_url, rating))
            course_map[course_url]["ratings"].append((student_url, rating))
        course_map[course_url]["average_rating"] = course_rating_total / len(review_info)
        total_rating += course_rating_total
        total_review += len(review_info)
    average_rating_all = total_rating / total_review

    for key in student_map:
        student_rating_total = 0
        for _, rating in student_map[key]["ratings"]:
            student_rating_total += rating
        student_map[key]["average_rating"] = student_rating_total / len(student_map[key]["ratings"])
        student_map[key]["rating_bias"] = student_map[key]["average_rating"] - average_rating_all

    for key in course_map:
        course_map[key]["rating_bias"] = course_map[key]["average_rating"] - average_rating_all

    i = 0
    for key in student_map:
        student_index_url[i] = key
        student_url_index[key] = i
        i += 1

    i = 0
    for key in course_map:
        course_index_url[i] = key
        course_url_index[key] = i
        i += 1

    d = 2
    alpha = 0.001
    lamda = 0.001
    iteration = 250
    print("Unfairness\t\tError\t\tValue\t\tAbsolute\t\tUnderestimation\t\tOverestimation\t\tNon-Parity")


    def helper(fairness_matrix=None):
        r = np.zeros((course_count, student_count))

        q = np.random.rand(course_count, d)
        p = np.random.rand(student_count, d)

        for course_url in course_map:
            row_index = course_url_index[course_url]
            for student_url, rating in course_map[course_url]["ratings"]:
                col_index = student_url_index[student_url]
                r[row_index][col_index] = rating

        matrix_name = "None"
        fairness = 0
        if fairness_matrix is not None:
            fairness = fairness_matrix(r, p, q) * 0.1
            matrix_name = fairness_matrix.__name__

        for _ in range(iteration):
            for course_url in course_map:
                row_index = course_url_index[course_url]
                for student_url, _ in course_map[course_url]["ratings"]:
                    col_index = student_url_index[student_url]
                    predict = float(np.dot(q[row_index], p[col_index]))

                    user_bias = student_map[student_url]["rating_bias"]
                    item_bias = course_map[course_url]["rating_bias"]

                    err = float(r[row_index][col_index]) - predict - user_bias - item_bias
                    tmp_q = q[row_index] + 2 * alpha * (
                                p[col_index] * err - lamda / 2 * q[row_index] - huber_loss(fairness) * q[row_index])
                    tmp_p = p[col_index] + 2 * alpha * (
                                q[row_index] * err - lamda / 2 * p[col_index] - huber_loss(fairness) * p[col_index])
                    q[row_index] = tmp_q
                    p[col_index] = tmp_p

                    course_map[course_url]["rating_bias"] = user_bias + 2 * alpha * (
                                err - lamda / 2 * user_bias - huber_loss(fairness) * user_bias)
                    student_map[student_url]["rating_bias"] = item_bias + 2 * alpha * (
                                err - lamda / 2 * item_bias - huber_loss(fairness) * item_bias)

        print(matrix_name + "\t\t" + str(total_err(r, p, q)) + "\t\t" + str(fair_val_all(r, p, q)) + "\t\t"
              + str(fair_abs_all(r, p, q)) + "\t\t" + str(fair_under_all(r, p, q)) + "\t\t"
              + str(fair_over_all(r, p, q)) + "\t\t" + str(fair_par(r, p, q)))


    helper(None)
    helper(fair_val_all)
    helper(fair_abs_all)
    helper(fair_under_all)
    helper(fair_over_all)
    helper(fair_par)

    # fair_val = np.zeros((course_count, 4))
    # fair_res = np.zeros((course_count, 1))
    # r = np.zeros((course_count, student_count))
    #
    # q = np.random.rand(course_count, d)
    # p = np.random.rand(student_count, d)
    #
    # for course_url in course_map:
    #     row_index = course_url_index[course_url]
    #     for student_url, rating in course_map[course_url]["ratings"]:
    #         col_index = student_url_index[student_url]
    #         r[row_index][col_index] = rating
    #
    # def t1(course_url):
    #     predict = e_g_item_predict(p, q, course_url)
    #     real = e_g_item_real(r, course_url)
    #     return predict, real
    # def t11():
    #     i = 0
    #     for course_url in course_map:
    #         vals = t1(course_url)
    #         fair_val[i][0] = vals[0][0]
    #         fair_val[i][1] = vals[0][1]
    #         fair_val[i][2] = vals[1][0]
    #         fair_val[i][3] = vals[1][1]
    #         fair_res[i][0] = abs(vals[0][0] - vals[1][0] - vals[0][1] + vals[1][1])
    #         i += 1
    #
    # t11()
    # total = 0
    # for i in range(course_count):
    #     total += fair_res[i][0]
    # print(total/course_count)

    # test purpose
