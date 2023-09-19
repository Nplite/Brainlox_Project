css = '''
<style>

.chat-message.user {
    border: 1px solid black;
    margin-left: 30rem;
    background-color: #89CFF0;
}

.chat-message.bot {
    border: 1px solid black;
    margin-right: 20rem;
    background-color: #B9D9EB;
    z-index:2 !important ;
    flex:1;
}



.chat-message .avatar {
  width: 20%;
}


.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}


.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem; display: flex
}



.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color:#000000;
}
 
.css-1vq4p4l h2 {
    font-size: 2rem;
    font-weight: 600;
}


.css-1y4p8pa {
    width: 100%;
    padding: 6rem 1rem 10rem ;
    max-width: 75rem !important;

    }

    
    .css-1avcm0n {
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 0rem;
    background: rgb(14, 17, 23);
    outline: none;
    z-index: 999990;
    display: block;
}


    
    .stTextInput {
    position: fixed;
    bottom: 0rem;
}


    .css-1xagxc0 {
        position: relative;
        width: 1120px;
        
    }


    .chat-message .avatar {
    width: 2%;
    }

.stTextInput {
    flex:2 ;
    z-index:1 !important ;
        }

.css-91z34k {
    width: 100%;
    padding: 6rem 1rem 10rem;
    max-width: 72rem;
}


.css-1544g2n h2 {
    font-size: 2rem;
    font-weight: 600;
}


'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''






