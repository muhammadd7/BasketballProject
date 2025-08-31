from django.urls import path
from myapp import views
from django.conf import settings
from django.conf.urls.static import static
import os


urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('get-results/', views.get_results, name='get_results'),  # This must match!
    path('search/', views.search_videos, name='search_videos'),
]

urlpatterns += static(settings.STATIC_URL, document_root=os.path.join(settings.BASE_DIR, 'myapp/static'))