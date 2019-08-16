from .test_ag_news import AgNewsTestCase


class YelpReviewFullTestCase(AgNewsTestCase):

    def setUp(self):
        super(YelpReviewFullTestCase, self).setUp()
        self.name = self.names[4]
        self.size = self.sizes[4]
