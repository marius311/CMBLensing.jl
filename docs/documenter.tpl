{% extends 'markdown.tpl'%}

{% block data_latex %}
der
{{ output.data['text/latex'] }}
{% endblock data_latex %}


{% block data_html scoped %}
```@raw html
{{ output.data['text/html'] }}
```
{% endblock data_html %}
