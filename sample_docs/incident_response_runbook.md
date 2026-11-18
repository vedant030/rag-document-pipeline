# Incident Response Runbook

Incidents are classified by severity. Severity 1 means a full outage or data loss affecting many customers. Severity 2 means a major feature is degraded. Severity 3 means a minor issue with a workaround.

When a Severity 1 incident is declared, the on-call engineer becomes the incident commander and opens a dedicated chat channel. The commander assigns a communications lead who posts customer updates on the status page every 30 minutes until the incident is resolved.

Responders should mitigate first and investigate later. Rolling back the most recent deploy is the preferred first mitigation, because most outages follow a change.

After every Severity 1 or Severity 2 incident we hold a blameless postmortem within five business days. The postmortem lists the timeline, root cause, customer impact and follow-up actions with owners and due dates.
