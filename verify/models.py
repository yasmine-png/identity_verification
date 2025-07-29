from djongo import models # type: ignore

class IdentityInfo(models.Model):
    photo_name = models.CharField(max_length=255)
    id_number = models.CharField(max_length=50)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id_number} - {self.first_name} {self.last_name}"
