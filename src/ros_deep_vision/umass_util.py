from perception_msgs.msg import State
from geometry_msgs.msg import Point, Pose

def to_state_pose_msg_list(value_dict, filter_xyz_dict):
    state_list = []
    pose_list = []
    for sig in value_dict:
        state = State()
        state.type = 'cnn'
        state.name = str(sig)
        state.value = value_dict[sig]
        state_list.append(state)

        pose = Pose()
        if not state.value == 0:
            pose.position.x = filter_xyz_dict[sig][0]
            pose.position.y = filter_xyz_dict[sig][1]
            pose.position.z = filter_xyz_dict[sig][2]
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 1
        pose_list.append(pose)
    return state_list, pose_list
