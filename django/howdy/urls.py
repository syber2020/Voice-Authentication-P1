# howdy/urls.py
from django.conf.urls import url
from howdy import views
from django.urls import path


urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),
    url(r'^RecordAudio/$', views.RecordAudio),
    url(r'^PredictEngine/$',views.PredictEngine),
    path("", views.RecordAudio, name="RecordAudio"),
    path("", views.PredictEngine, name="PredictEngine"),
]
