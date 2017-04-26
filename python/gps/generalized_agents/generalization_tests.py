from __future__ import print_function
from reacher_by_color_and_type import reacher_by_color_and_type, RobotType
from gps.example_agents.reach_3link_black import reach_3link_black
from gps.example_agents.reach_3link_red import reach_3link_red
from gps.example_agents.reach_3link_green import reach_3link_green
from gps.example_agents.reach_3link_yellow import reach_3link_yellow
from gps.example_agents.reach_3link_black_shortjoint import reach_3link_black_shortjoint
from gps.example_agents.reach_3link_red_shortjoint import reach_3link_red_shortjoint
from gps.example_agents.reach_3link_green_shortjoint import reach_3link_green_shortjoint
from gps.example_agents.reach_3link_yellow_shortjoint import reach_3link_yellow_shortjoint
from gps.example_agents.reach_4link_red import reach_4link_red
from gps.example_agents.reach_4link_black import reach_4link_black
from gps.example_agents.reach_4link_green import reach_4link_green
from gps.example_agents.reach_4link_yellow import reach_4link_yellow

import numpy as np

old = {
    RobotType.THREE_LINK_SHORT_JOINT : {"black" : reach_3link_black_shortjoint, "red" : reach_3link_red_shortjoint, "yellow" : reach_3link_yellow_shortjoint, "green" : reach_3link_green_shortjoint},
    RobotType.THREE_LINK : {"black" : reach_3link_black, "red" : reach_3link_red, "yellow" : reach_3link_yellow, "green" : reach_3link_green},
    RobotType.FOUR_LINK : {"black" : reach_4link_black, "red" : reach_4link_red, "yellow" : reach_4link_yellow, "green": reach_4link_green }
}

for num_robots in (10, 20, 30):
    for robot_num in np.random.choice(num_robots, size=10):
        for typ in old:
            for color in old[typ]:
                if typ == RobotType.FOUR_LINK and color == "black":
                    offsets = [[-0.4, 0.0, 0.7], [0.45, 0.0, 0.45], [0.4, 0.0, -1.3], [-0.3, 0.0, -1.65]]
                else:
                    offsets = [[-0.3, 0., -1.65], [0.4, 0., -1.3],  [0.45, 0., 0.45], [-0.4, 0.0, 0.7]]
                generic = reacher_by_color_and_type(num_robots, robot_num, [0.8, 0.0, 0.5], offsets, color, typ, enable_images=True)
                specialized = old[typ][color](num_robots, robot_num)
                specialized['agent']['pos_body_offset'] = None # Works!
                np.testing.assert_equal(generic, specialized, "num_robots=%s, robot_num=%s, color=%s, typ=%s" % (num_robots, robot_num, color, typ))


print("All tests passed!")
