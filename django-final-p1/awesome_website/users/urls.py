# users/urls.py

from django.conf.urls import include, url
from users.views import dashboard, register
from users import views
from django.urls import path
urlpatterns = [
    url(r"^accounts/", include("django.contrib.auth.urls")),
    url(r"^dashboard/", dashboard, name="dashboard"),
    url(r"^register/", register, name="register"),
    url(r'^$', views.HomePageView.as_view()),
    url(r'^RecordAudio/$', views.RecordAudio),
    url(r'^PredictEngine/$',views.PredictEngine),
    path("", views.RecordAudio, name="RecordAudio"),
    path("", views.PredictEngine, name="PredictEngine"),

]
