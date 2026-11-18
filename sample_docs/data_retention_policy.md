# Data Retention Policy

Customer account data is kept for as long as the account is active. After an account is closed, we retain customer data for 90 days so that accidental closures can be reversed, and then it is permanently deleted from production systems.

Backups are encrypted and stored in a separate region. Backup snapshots are kept for 35 days and then expire automatically. Deleted data may therefore remain in backups until the snapshot that contains it expires.

Application logs that contain personal data are retained for 30 days. Aggregated usage metrics that contain no personal data may be kept for up to 24 months for capacity planning.

Customers can request deletion of their data at any time by contacting support. Deletion requests are completed within 30 days, and we confirm completion by email. Legal holds override all deletion schedules until the hold is lifted.
