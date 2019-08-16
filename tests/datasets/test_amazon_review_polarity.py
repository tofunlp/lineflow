from .test_ag_news import AgNewsTestCase


class AmazonReviewPolarityTestCase(AgNewsTestCase):

    def setUp(self):
        super(AmazonReviewPolarityTestCase, self).setUp()
        self.name = self.names[6]
        self.size = self.sizes[6]
