# Generated by Django 4.2.7 on 2025-07-18 23:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('APIapp', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='books',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='books',
            name='updated_at',
        ),
        migrations.AlterField(
            model_name='books',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]
