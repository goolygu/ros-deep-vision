#! /usr/bin/env python
from data_settings import *
import settings
import yaml

if __name__ == '__main__':
    tbp = True#False#
    ds = DataSettings(tbp)
    name = ds.get_name() + "_" + ds.get_test_name()

    if tbp:
        path = settings.ros_dir + '/data/'
    else:
        path = settings.ros_dir + '/data_notbp/'

    f = open(path + "/result/cross_validation_" + name + '.yaml')
    result = yaml.load(f)

    err_sum = {}
    count = {}

    for case in result:
        if not case in err_sum:
            err_sum[case] = {}
            count[case] = 0
        for obj in result[case]:
            for frame in result[case][obj]:
                if not frame in err_sum[case]:
                    err_sum[case][frame] = 0
                err_sum[case][frame] += result[case][obj][frame]
            count[case] += 1

    for case in err_sum:
        for frame in err_sum[case]:
            print "case", case, "frame", frame, "avg", err_sum[case][frame]/count[case]

    print

    for case in err_sum:
        frame_err_sum = 0
        frame_count = 0
        for frame in err_sum[case]:
            frame_err_sum += err_sum[case][frame]
            frame_count += 1

        print "case", case, "avg", frame_err_sum/(frame_count*count[case])
