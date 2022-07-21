
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    # url(r'^admin/', admin.site.urls),
    # url(r'^$', views.home),
    path('',views.home,name='home'),
    path('ourTeam/',views.ourTeam,name='ourTeam'),
    path('about/',views.about,name='about'),
    path('modification/',views.modification,name='modification'),
    path('improvements/',views.improvements,name='improvements'),
    path('mlModel/', views.mlModel,name='ml_model'),
    path('ocrModel/', views.ocrModel,name='ocr_model'),

    
]