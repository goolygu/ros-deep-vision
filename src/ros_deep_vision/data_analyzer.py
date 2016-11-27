#! /usr/bin/env python
from data_settings import *
import settings
import yaml
import numpy as np

def check_fail(result_fail):
    for case in result_fail:
        for obj in result_fail[case]:
            for frame in result_fail[case][obj]:
                if result_fail[case][obj][frame] > 0:
                    print "FAILED"
                    print result_fail
                    return
    print "SUCCEEDED"

def run_analysis(result):
    # a dictionary with [case][frame]
    err_sum = {}
    obj_count = {}

    fail_count = 0
    finger_threshold = 0.05
    palm_threshold = 0.1
    total_obj = 0

    print_latex = True

    if print_latex == True:
        for case in result:
            for obj in result[case]:
                print obj, "&",
            print

            obj1 = result[case].keys()[0]
            for frame in result[case][obj1]:
                for obj in result[case]:
                    print round(result[case][obj][frame] ,4), "&",

                print
            print


    for case in result:
        if not case in err_sum:
            err_sum[case] = {}
            obj_count[case] = 0
        for obj in result[case]:
            # for frame in result[case][obj]:
            #     if not frame in err_sum[case]:
            #         err_sum[case][frame] = 0
            #     err_sum[case][frame] += result[case][obj][frame]
            #     print case, obj, frame, result[case][obj][frame]

            failed = False
            for frame in result[case][obj]:
                print case, obj, frame, result[case][obj][frame]


                if not frame == "r2/left_palm" and result[case][obj][frame] > finger_threshold:
                    fail_count += 1
                    failed = True
                    break
                elif frame == "r2/left_palm" and result[case][obj][frame] > palm_threshold:
                    fail_count += 1
                    failed = True
                    break
                elif np.isnan(result[case][obj][frame]):
                    fail_count += 1
                    failed = True
                    break

            # only count if didn't fail
            if failed == False:
                for frame in result[case][obj]:
                    if not frame in err_sum[case]:
                        err_sum[case][frame] = 0
                    err_sum[case][frame] += result[case][obj][frame]

                obj_count[case] += 1
            total_obj += 1
    print " "

    for case in err_sum:
        for frame in err_sum[case]:
            print "case", case, "frame", frame, "avg", err_sum[case][frame]/obj_count[case]

    print " "

    all_err_sum = 0
    all_err_sum_no_palm = 0
    all_count = 0
    all_count_no_palm = 0



    for case in err_sum:
        frame_err_sum = 0
        frame_count = 0

        for frame in err_sum[case]:
            frame_err_sum += err_sum[case][frame]
            frame_count += 1
            all_err_sum += err_sum[case][frame]
            all_count += obj_count[case]
            if frame != "r2/left_palm":
                all_err_sum_no_palm += err_sum[case][frame]
                all_count_no_palm += obj_count[case]
        # all_count += frame_count*obj_count[case]

        if frame_count*obj_count[case] > 0:
            case_avg_err = frame_err_sum/(frame_count*obj_count[case])
            print "case", case, "avg", case_avg_err
        else:
            print "case", case, "failed"

    print " "

    print "failed", fail_count, "/", total_obj

    if all_count_no_palm > 0:
        print "no palm avg", all_err_sum_no_palm/all_count_no_palm
    if all_count > 0:
        print "overall avg", all_err_sum/all_count

if __name__ == '__main__':
    tbp = True#False#
    ds = DataSettings()
    name = ds.get_name() + "_" + ds.get_test_name()

    path = settings.ros_dir
    case = 1
    if case == 1:
        f = open(path + "/result/cross_validation_fail_" + name + '.yaml')
        result_fail = yaml.load(f)
        check_fail(result_fail)
        f = open(path + "/result/cross_validation_" + name + '.yaml')
        # a dictionary with [case][object][frame]
        result = yaml.load(f)
        run_analysis(result)

    elif case == 2:
        f = open(path + "/result/" + name + '.yaml')
        # a dictionary with [case][object][frame]
        result = yaml.load(f)
        run_analysis(result)
