# Generated by Django 5.1 on 2024-09-28 09:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('appointment', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='medical_history',
            field=models.CharField(max_length=100),
        ),
    ]