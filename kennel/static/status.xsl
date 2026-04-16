<?xml version="1.0" encoding="UTF-8"?>
<!--
  Rube Goldberg layer 2: structural XML → display XML.

  Input:  structural XML in xmlns="https://fidocancode.dog/kennel"
          with dog:status attributes from xmlns:dog="https://fidocancode.dog/woof"
  Output: display XML in xmlns="https://fidocancode.dog/display"
          with dog:status attributes preserved, plus a CSS PI so the browser
          styles the result via status.css

  Pipeline: server → structural XML → [this XSLT] → display XML → CSS
-->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:k="https://fidocancode.dog/kennel"
  xmlns:dog="https://fidocancode.dog/woof"
  xmlns="https://fidocancode.dog/display"
  exclude-result-prefixes="k">

<xsl:output method="xml" encoding="UTF-8" indent="yes"/>

<xsl:template match="/">
  <xsl:processing-instruction name="xml-stylesheet">type="text/css" href="/static/status.css"</xsl:processing-instruction>
  <dashboard>
    <title>&#x1F43E; kennel</title>
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

<xsl:template match="k:repo">
  <card>
    <xsl:attribute name="dog:status">
      <xsl:value-of select="@dog:status"/>
    </xsl:attribute>

    <name><xsl:value-of select="k:repo_name"/></name>

    <xsl:if test="k:worker_uptime_seconds != ''">
      <uptime>
        <xsl:text>up </xsl:text>
        <xsl:call-template name="format-duration">
          <xsl:with-param name="seconds" select="k:worker_uptime_seconds"/>
        </xsl:call-template>
      </uptime>
    </xsl:if>

    <activity><xsl:value-of select="k:what"/></activity>

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
    </badges>

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

    <xsl:if test="k:last_crash_error != ''">
      <crash-detail><xsl:value-of select="k:last_crash_error"/></crash-detail>
    </xsl:if>

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

<!-- Format seconds into compact human-readable duration -->
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

</xsl:stylesheet>
