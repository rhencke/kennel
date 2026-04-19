<?xml version="1.0" encoding="UTF-8"?>
<!--
  Rube Goldberg layer 2: structural XML → display XML.

  Input:  structural XML in xmlns="https://fidocancode.dog/kennel"
          with dog:status attributes from xmlns:dog="https://fidocancode.dog/woof"
  Output: display XML in xmlns="https://fidocancode.dog/display"
          with dog:status attributes preserved, plus a CSS PI so the browser
          styles the result via status.css

  Pipeline: server → structural XML → [this XSLT] → display XML → CSS

  Root <kennel> element layout:
    <kennel_uptime_seconds>  — seconds since kennel started
    <rate_limit>             — GitHub rate-limit snapshot
      <rest> <used> <limit> <resets_at>
      <graphql> ...
    <repo dog:status="...">  — one per registered repo
      scalar fields (repo_name, what, busy, ...)
      <provider_status> nested dict
      <claude_talker> nested dict
      <issue_cache> nested dict
      <webhook_activities> list of <webhook>
-->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:k="https://fidocancode.dog/kennel"
  xmlns:dog="https://fidocancode.dog/woof"
  xmlns="https://fidocancode.dog/display"
  exclude-result-prefixes="k">

<xsl:output method="xml" encoding="UTF-8" indent="yes"/>

<!-- ═══════════════════════════════════════════════════════════════════════
     Root template — dashboard wrapper
     ═══════════════════════════════════════════════════════════════════════ -->

<xsl:template match="/">
  <xsl:processing-instruction name="xml-stylesheet">type="text/css" href="/static/status.css"</xsl:processing-instruction>
  <dashboard>
    <title>&#x1F43E; kennel</title>

    <!-- Dashboard header: kennel uptime + GitHub rate limits -->
    <xsl:if test="k:kennel/k:kennel_uptime_seconds != '' or k:kennel/k:rate_limit/k:rest/k:used != ''">
      <header>
        <xsl:if test="k:kennel/k:kennel_uptime_seconds != ''">
          <kennel-uptime>
            <xsl:text>up </xsl:text>
            <xsl:call-template name="format-duration">
              <xsl:with-param name="seconds" select="k:kennel/k:kennel_uptime_seconds"/>
            </xsl:call-template>
          </kennel-uptime>
        </xsl:if>
        <xsl:if test="k:kennel/k:rate_limit/k:rest/k:used != ''">
          <rate-limits>
            <rate-window dog:kind="rest">
              <rate-label>REST</rate-label>
              <rate-nums>
                <xsl:value-of select="k:kennel/k:rate_limit/k:rest/k:used"/>
                <xsl:text>/</xsl:text>
                <xsl:value-of select="k:kennel/k:rate_limit/k:rest/k:limit"/>
              </rate-nums>
              <xsl:if test="k:kennel/k:rate_limit/k:rest/k:resets_at != ''">
                <rate-reset>
                  <xsl:text>resets </xsl:text>
                  <xsl:call-template name="format-time">
                    <xsl:with-param name="iso" select="k:kennel/k:rate_limit/k:rest/k:resets_at"/>
                  </xsl:call-template>
                </rate-reset>
              </xsl:if>
            </rate-window>
            <xsl:if test="k:kennel/k:rate_limit/k:graphql/k:used != ''">
              <rate-window dog:kind="graphql">
                <rate-label>GraphQL</rate-label>
                <rate-nums>
                  <xsl:value-of select="k:kennel/k:rate_limit/k:graphql/k:used"/>
                  <xsl:text>/</xsl:text>
                  <xsl:value-of select="k:kennel/k:rate_limit/k:graphql/k:limit"/>
                </rate-nums>
                <xsl:if test="k:kennel/k:rate_limit/k:graphql/k:resets_at != ''">
                  <rate-reset>
                    <xsl:text>resets </xsl:text>
                    <xsl:call-template name="format-time">
                      <xsl:with-param name="iso" select="k:kennel/k:rate_limit/k:graphql/k:resets_at"/>
                    </xsl:call-template>
                  </rate-reset>
                </xsl:if>
              </rate-window>
            </xsl:if>
          </rate-limits>
        </xsl:if>
      </header>
    </xsl:if>

    <xsl:choose>
      <xsl:when test="k:kennel/k:repo">
        <xsl:apply-templates select="k:kennel/k:repo"/>
      </xsl:when>
      <xsl:otherwise>
        <empty>No repos configured. Napping &#x1F4A4;</empty>
      </xsl:otherwise>
    </xsl:choose>
  </dashboard>
</xsl:template>

<!-- ═══════════════════════════════════════════════════════════════════════
     Per-repo card
     ═══════════════════════════════════════════════════════════════════════ -->

<xsl:template match="k:repo">
  <card>
    <xsl:attribute name="dog:status">
      <xsl:value-of select="@dog:status"/>
    </xsl:attribute>

    <!-- ── Name + fido state ── -->
    <name>
      <xsl:value-of select="k:repo_name"/>
    </name>

    <!-- Worker uptime floated right -->
    <xsl:if test="k:worker_uptime_seconds != ''">
      <uptime>
        <xsl:text>up </xsl:text>
        <xsl:call-template name="format-duration">
          <xsl:with-param name="seconds" select="k:worker_uptime_seconds"/>
        </xsl:call-template>
      </uptime>
    </xsl:if>

    <!-- Current worker activity text -->
    <activity>
      <xsl:value-of select="k:what"/>
    </activity>

    <!-- ── Issue assignment ── -->
    <xsl:if test="k:issue != ''">
      <issue-row>
        <issue-num>
          <xsl:text>#</xsl:text>
          <xsl:value-of select="k:issue"/>
        </issue-num>
        <xsl:if test="k:issue_title != ''">
          <issue-title>
            <xsl:value-of select="k:issue_title"/>
          </issue-title>
        </xsl:if>
        <xsl:if test="k:issue_elapsed_seconds != ''">
          <issue-elapsed>
            <xsl:call-template name="format-duration">
              <xsl:with-param name="seconds" select="k:issue_elapsed_seconds"/>
            </xsl:call-template>
          </issue-elapsed>
        </xsl:if>
      </issue-row>
    </xsl:if>

    <!-- ── PR assignment ── -->
    <xsl:if test="k:pr_number != ''">
      <pr-row>
        <pr-num>
          <xsl:text>PR #</xsl:text>
          <xsl:value-of select="k:pr_number"/>
        </pr-num>
        <xsl:if test="k:pr_title != ''">
          <pr-title>
            <xsl:value-of select="k:pr_title"/>
          </pr-title>
        </xsl:if>
      </pr-row>
    </xsl:if>

    <!-- ── Current task progress ── -->
    <xsl:if test="k:current_task != ''">
      <task-row>
        <xsl:if test="k:task_number != '' and k:task_total != ''">
          <task-pos>
            <xsl:value-of select="k:task_number"/>
            <xsl:text>/</xsl:text>
            <xsl:value-of select="k:task_total"/>
          </task-pos>
        </xsl:if>
        <task-title>
          <xsl:value-of select="k:current_task"/>
        </task-title>
      </task-row>
    </xsl:if>

    <!-- ── Badges ── -->
    <badges>
      <xsl:if test="k:crash_count &gt; 0">
        <badge dog:kind="crash">
          <xsl:value-of select="k:crash_count"/>
          <xsl:text> crash</xsl:text>
          <xsl:if test="k:crash_count &gt; 1">es</xsl:if>
        </badge>
      </xsl:if>

      <xsl:if test="k:rescoping = 'true'">
        <badge dog:kind="rescope">rescoping &#x27F3;</badge>
      </xsl:if>

      <xsl:if test="k:session_alive = 'true'">
        <badge dog:kind="session">
          <xsl:text>session</xsl:text>
          <xsl:if test="k:session_owner != ''">
            <xsl:text>: </xsl:text>
            <xsl:value-of select="k:session_owner"/>
          </xsl:if>
          <xsl:if test="k:session_pid != ''">
            <xsl:text> (pid </xsl:text>
            <xsl:value-of select="k:session_pid"/>
            <xsl:text>)</xsl:text>
          </xsl:if>
        </badge>
      </xsl:if>

      <xsl:if test="k:session_dropped_count &gt; 0">
        <badge dog:kind="session-dropped">
          <xsl:value-of select="k:session_dropped_count"/>
          <xsl:text> session</xsl:text>
          <xsl:if test="k:session_dropped_count &gt; 1">s</xsl:if>
          <xsl:text> dropped</xsl:text>
        </badge>
      </xsl:if>

      <xsl:if test="k:provider_status/k:paused = 'true'">
        <badge dog:kind="provider-paused">
          <xsl:value-of select="k:provider"/>
          <xsl:text> paused</xsl:text>
          <xsl:if test="k:provider_status/k:resets_at != ''">
            <xsl:text> until </xsl:text>
            <xsl:call-template name="format-time">
              <xsl:with-param name="iso" select="k:provider_status/k:resets_at"/>
            </xsl:call-template>
          </xsl:if>
        </badge>
      </xsl:if>
    </badges>

    <!-- ── Provider pressure (when not paused) ── -->
    <xsl:if test="k:provider_status/k:percent_used != '' and k:provider_status/k:paused != 'true'">
      <pressure>
        <pressure-label>
          <xsl:value-of select="k:provider"/>
        </pressure-label>
        <pressure-pct>
          <xsl:value-of select="k:provider_status/k:percent_used"/>
          <xsl:text>%</xsl:text>
        </pressure-pct>
        <xsl:if test="k:provider_status/k:window_name != ''">
          <pressure-window>
            <xsl:value-of select="k:provider_status/k:window_name"/>
          </pressure-window>
        </xsl:if>
      </pressure>
    </xsl:if>

    <!-- ── Active claude talker ── -->
    <xsl:if test="k:claude_talker/k:kind">
      <talker>
        <label>claude</label>
        <kind><xsl:value-of select="k:claude_talker/k:kind"/></kind>
        <desc><xsl:value-of select="k:claude_talker/k:description"/></desc>
        <xsl:if test="k:claude_talker/k:claude_pid != ''">
          <pid>
            <xsl:text>pid </xsl:text>
            <xsl:value-of select="k:claude_talker/k:claude_pid"/>
          </pid>
        </xsl:if>
      </talker>
    </xsl:if>

    <!-- ── Last crash error ── -->
    <xsl:if test="k:last_crash_error != ''">
      <crash-detail><xsl:value-of select="k:last_crash_error"/></crash-detail>
    </xsl:if>

    <!-- ── Issue cache summary ── -->
    <xsl:if test="k:issue_cache/k:loaded = 'true'">
      <issue-cache>
        <xsl:if test="k:issue_cache/k:open_issues != ''">
          <cache-issues>
            <xsl:value-of select="k:issue_cache/k:open_issues"/>
            <xsl:text> open</xsl:text>
          </cache-issues>
        </xsl:if>
        <xsl:if test="k:issue_cache/k:events_applied &gt; 0">
          <cache-events>
            <xsl:value-of select="k:issue_cache/k:events_applied"/>
            <xsl:text> event</xsl:text>
            <xsl:if test="k:issue_cache/k:events_applied &gt; 1">s</xsl:if>
            <xsl:text> applied</xsl:text>
          </cache-events>
        </xsl:if>
        <xsl:if test="k:issue_cache/k:events_dropped_stale &gt; 0">
          <cache-dropped>
            <xsl:value-of select="k:issue_cache/k:events_dropped_stale"/>
            <xsl:text> stale</xsl:text>
          </cache-dropped>
        </xsl:if>
      </issue-cache>
    </xsl:if>

    <!-- ── In-flight webhook handlers ── -->
    <xsl:if test="k:webhook_activities/k:webhook">
      <hooks>
        <hooks-label>webhooks</hooks-label>
        <xsl:for-each select="k:webhook_activities/k:webhook">
          <hook>
            <hook-desc><xsl:value-of select="k:description"/></hook-desc>
            <elapsed>
              <xsl:call-template name="format-duration">
                <xsl:with-param name="seconds" select="k:elapsed_seconds"/>
              </xsl:call-template>
            </elapsed>
          </hook>
        </xsl:for-each>
      </hooks>
    </xsl:if>

  </card>
</xsl:template>

<!-- ═══════════════════════════════════════════════════════════════════════
     Utility templates
     ═══════════════════════════════════════════════════════════════════════ -->

<!-- Format seconds into compact human-readable duration: 2h13m, 45m, 8s -->
<xsl:template name="format-duration">
  <xsl:param name="seconds"/>
  <xsl:variable name="s" select="floor(number($seconds))"/>
  <xsl:variable name="h" select="floor($s div 3600)"/>
  <xsl:variable name="m" select="floor(($s mod 3600) div 60)"/>
  <xsl:variable name="sec" select="$s mod 60"/>
  <xsl:choose>
    <xsl:when test="$h &gt; 0 and $m &gt; 0">
      <xsl:value-of select="$h"/><xsl:text>h</xsl:text>
      <xsl:value-of select="$m"/><xsl:text>m</xsl:text>
    </xsl:when>
    <xsl:when test="$h &gt; 0">
      <xsl:value-of select="$h"/><xsl:text>h</xsl:text>
    </xsl:when>
    <xsl:when test="$m &gt; 0">
      <xsl:value-of select="$m"/><xsl:text>m</xsl:text>
    </xsl:when>
    <xsl:otherwise>
      <xsl:value-of select="$sec"/><xsl:text>s</xsl:text>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<!-- Extract HH:MM UTC from an ISO-8601 timestamp (e.g. 2026-04-19T13:00:00+00:00) -->
<xsl:template name="format-time">
  <xsl:param name="iso"/>
  <xsl:variable name="time-part" select="substring-after($iso, 'T')"/>
  <xsl:value-of select="substring($time-part, 1, 5)"/>
  <xsl:text> UTC</xsl:text>
</xsl:template>

</xsl:stylesheet>
