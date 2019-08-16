from .test_ag_news import AgNewsTestCase


class YelpReviewPolarityTestCase(AgNewsTestCase):

    def setUp(self):
        super(YelpReviewPolarityTestCase, self).setUp()
        self.name = self.names[3]
        self.size = self.sizes[3]
