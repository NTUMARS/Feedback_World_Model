<h1 align="center">Feedback World Model Enables<br />Precise Guidance of Diffusion Policy</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.15705"><img src="https://img.shields.io/badge/arXiv-2605.15705-b31b1b?logo=arxiv&logoColor=white" alt="arXiv" /></a>
  <a href="https://arxiv.org/pdf/2605.15705"><img src="https://img.shields.io/badge/Paper-PDF-d32f2f?logo=adobeacrobatreader&logoColor=white" alt="Paper PDF" /></a>
  <a href="https://ntumars.github.io/Feedback_World_Model/"><img src="https://img.shields.io/badge/Project_Page-Live-2ea44f?logo=googlechrome&logoColor=white" alt="Project Page" /></a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=Lorenzo-0-0.Feedback_World_Model&left_text=Visitors&right_color=blueviolet" alt="Visitors" />
</p>

<p align="center">
  <a href="https://morpheus-an.github.io/"><strong>Tuo An</strong></a><sup>1,*</sup> &nbsp;&nbsp;
  <a href="https://jiajindou.github.io/"><strong>Jindou Jia</strong></a><sup>1,*</sup> &nbsp;&nbsp;
  <a href="https://reagan1311.github.io/"><strong>Gen Li</strong></a><sup>1</sup> &nbsp;&nbsp;
  <a href="https://lorenzo-0-0.github.io/"><strong>Jingliang Li</strong></a><sup>1</sup> &nbsp;&nbsp;
  <a href="https://chuhaozhou99.github.io/Chuhao-Zhou/"><strong>Chuhao Zhou</strong></a><sup>1</sup>
  <br />
  <strong>Pengfei Liu</strong><sup>1</sup> &nbsp;&nbsp;
  <strong>Bofan Lyu</strong><sup>1</sup> &nbsp;&nbsp;
  <strong>Jiaqi Bai</strong><sup>1</sup> &nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?hl=en&user=KWXEabIAAAAJ&view_op=list_works&sortby=pubdate"><strong>Xinying Guo</strong></a><sup>1</sup> &nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=SZg1xeoAAAAJ&hl=zh-CN"><strong>Geng Li</strong></a><sup>1</sup> &nbsp;&nbsp;
  <a href="https://marsyang.site/"><strong>Jianfei Yang</strong></a><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>MARS Lab, Nanyang Technological University, Singapore
</p>

<p align="center">
  <sup>*</sup>Equal contribution &nbsp;&nbsp;&middot;&nbsp;&nbsp; <sup>†</sup>Corresponding author
</p>

<p align="center">
  <img src="./static/images/teaser.png" alt="Feedback-Guided Policy architecture" width="95%" />
</p>

<table align="center" width="90%">
  <tr>
    <td valign="top">
      <b>Abstract.</b> World models aim to improve robotic decision making by predicting the consequences of actions. However, in practice, their predictions often become unreliable once the robot encounters states outside the training distribution, limiting their effectiveness at deployment. We observe that execution itself provides a natural but underutilized signal: after each action, the robot directly observes the true next state, revealing the mismatch between predicted and actual outcomes. Building on this insight, we propose the <b>feedback world model</b>, a new paradigm that closes the loop between prediction and observation at inference time. Instead of treating the world model as a static open-loop predictor, our method maintains a lightweight feedback state that is updated online to iteratively correct future predictions, compensating for model errors using real-time observations <b>without additional training data or parameter updates</b>. We show that this process can be interpreted as a <b>latent-space observer</b> and admits convergence guarantees under mild conditions. We further introduce <b>action-aware guidance</b> to better translate corrected predictions into control by emphasizing action-controllable components while suppressing irrelevant variations. Experiments on LIBERO-Plus, Robomimic, and real-world manipulation tasks demonstrate that our method substantially improves both prediction accuracy and policy performance under distribution shift. In particular, it reduces world model prediction error by up to <b>76.4%</b> and improves out-of-distribution (OOD) success rate by <b>30%</b>.
      <br /><br />
      <img src="./static/images/mars_lab_logo.png" alt="MARS Lab" width="110" align="right" />
      <b>Correspondence:</b> Jianfei Yang &lt;<a href="mailto:jianfei.yang@ntu.edu.sg">jianfei.yang@ntu.edu.sg</a>&gt;
    </td>
  </tr>
</table>
