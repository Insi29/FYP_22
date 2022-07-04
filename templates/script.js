$(document).ready(function(){
    
	var pakainfoEvent = false;
	$('#bootstrapNewsSlider').carousel({
		interval:   4000	
	}).on('click', '.list-group li', function() {
			pakainfoEvent = true;
			$('.list-group li').removeClass('active');
			$(this).addClass('active');		
	}).on('slid.bs.carousel', function(e) {
		if(!pakainfoEvent) {
			var count = $('.list-group').children().length -1;
			var current = $('.list-group li.active');
			current.removeClass('active').next().addClass('active');
			var id = parseInt(current.data('slide-to'));
			if(count == id) {
				$('.list-group li').first().addClass('active');	
			}
		}
		pakainfoEvent = false;
	});
})

$(window).load(function() {
    var boxheight = $('#bootstrapNewsSlider .carousel-inner').innerHeight();
    var itemlength = $('#bootstrapNewsSlider .item').length;
    var triggerheight = Math.round(boxheight/itemlength+1);
	$('#bootstrapNewsSlider .list-group-item').outerHeight(triggerheight);
});