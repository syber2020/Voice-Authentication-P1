#users/forms.py

# from django.contrib.auth.forms import UserCreationForm
# from django.contrib.auth.models import User


# class CustomUserCreationForm(UserCreationForm):
#     First_Name = forms.CharField(max_length=100)
#     Last_Name = forms.CharField(max_length=100)
#     class Meta(UserCreationForm.Meta):
#         fields = UserCreationForm.Meta.fields + ("email","First Name","Last Name",)

from django.contrib.auth.forms import UserCreationForm

class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        fields = UserCreationForm.Meta.fields + ("email",)

# from django.contrib.auth.forms import UserCreationForm
# from django import forms
# from django.contrib.auth.models import User
# from django.contrib.auth import login, authenticate
# # class CustomUserCreationForm(UserCreationForm):
# #     username = forms.CharField(max_length=20)
# #     email = forms.EmailField(label = "Email")
# #     fullname = forms.CharField(label = "Name")
# #     address = forms.CharField(label = "Address")
# #     state = forms.CharField(label = "State")

# #     class Meta:
# #         model = User
# #         fields = ( "fullname", "email","address","state","username" )

# def CustomUserCreationForm(request):
#     if request.method == 'POST':
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             form.save()
#             username = form.cleaned_data.get('username')
#             raw_password = form.cleaned_data.get('password1')
#             user = authenticate(username=username,password=raw_password)
#             email = forms.EmailField(label="Email")
#      		fullname = forms.CharField(label="Name")
#      		address = forms.CharField(label="Address")
#      		state = forms.CharField(label="State")
#             login(request, user)
#             return redirect('home')
#         else:
#             form = UserCreationForm()
#     return render(request, 'dashboard.html', {'form': form})