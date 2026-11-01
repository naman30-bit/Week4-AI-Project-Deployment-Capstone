import unittest
from app import app
class ApiTests(unittest.TestCase):
 def setUp(self): self.client=app.test_client()
 def test_home(self): self.assertEqual(self.client.get("/").status_code,200)
 def test_health(self): self.assertEqual(self.client.get("/health").status_code,200)
 def test_prediction(self):
  r=self.client.post("/predict",json={"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2})
  self.assertEqual(r.status_code,200); self.assertIn("predicted_class",r.get_json())
 def test_missing_input(self): self.assertEqual(self.client.post("/predict",json={"sepal length (cm)":5.1}).status_code,400)
if __name__=="__main__": unittest.main()
