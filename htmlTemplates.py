css = """
<style>
    .justified {
        text-align: justify;
        font-size: 16px;
        line-height: 1.5;
    }
    .chat-bubble {
        border-radius: 12px;
        padding: 12px;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.6;
        text-align: justify;
    }
</style>
"""

user_template = """
<div class="chat-bubble" style="background-color: #DCF8C6;"><strong>User:</strong><br>{{MSG}}</div>
"""

bot_template = """
<div class="chat-bubble" style="background-color: #F1F0F0;"><strong>Bot:</strong><br>{{MSG}}</div>
"""
