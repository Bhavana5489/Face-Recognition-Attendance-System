import unittest

class TestDotPositions(unittest.TestCase):
    def test_1280_720_resolution(self):
        w_frame = 1280
        h_frame = 720
        
        # Test coordinates mapping
        # LEFT: screen_x = 25, screen_y = 50
        xd_left = int(25 * w_frame / 100.0)
        yd_left = int(50 * h_frame / 100.0)
        self.assertEqual(xd_left, 320)
        self.assertEqual(yd_left, 360)
        
        # RIGHT: screen_x = 75, screen_y = 50
        xd_right = int(75 * w_frame / 100.0)
        yd_right = int(50 * h_frame / 100.0)
        self.assertEqual(xd_right, 960)
        self.assertEqual(yd_right, 360)
        
        # UP: screen_x = 50, screen_y = 25
        xd_up = int(50 * w_frame / 100.0)
        yd_up = int(25 * h_frame / 100.0)
        self.assertEqual(xd_up, 640)
        self.assertEqual(yd_up, 180)
        
        # DOWN: screen_x = 50, screen_y = 75
        xd_down = int(50 * w_frame / 100.0)
        yd_down = int(75 * h_frame / 100.0)
        self.assertEqual(xd_down, 640)
        self.assertEqual(yd_down, 540)
        
        # CENTER/BLINK: screen_x = 50, screen_y = 50
        xd_center = int(50 * w_frame / 100.0)
        yd_center = int(50 * h_frame / 100.0)
        self.assertEqual(xd_center, 640)
        self.assertEqual(yd_center, 360)

if __name__ == "__main__":
    unittest.main()
